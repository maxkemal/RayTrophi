"""Webcam or video file -> MediaPipe multi-pose -> room joints -> external Player IPC.

Frames stay in memory. No preview, image encoding, recording or landmark logs.
Only aggregate timing/error counts may be written to --metrics.
"""
import argparse
import json
import math
from pathlib import Path
import time
import sys
import threading

try:
    from .calibration import GroundCalibration
    from .schema import validate_pose
    from .tracking import PersonTracker
    from .transport import PIPE, connect, summary_ms, wait_frame
    from .video_card import VideoCardPublisher, cutout, mask_input
except ImportError:
    from calibration import GroundCalibration
    from schema import validate_pose
    from tracking import PersonTracker
    from transport import PIPE, connect, summary_ms, wait_frame
    from video_card import VideoCardPublisher, cutout, mask_input

LANDMARKS = {"head": 0, "l_shoulder": 11, "r_shoulder": 12,
             "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
             "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
             "l_ankle": 27, "r_ankle": 28}


def confidence(point):
    return max(0.0, min(1.0, float(min(point.visibility, point.presence))))


def convert_pose(image_points, world_points, calibration, min_confidence=0.5):
    """Ground feet anchor + metric hip-relative MP skeleton (approximate yaw).

    Homography is valid on the floor only. Do not project hands through it.
    Reject missing feet rather than moving the person to a made-up origin.
    """
    if len(image_points) != 33 or len(world_points) != 33:
        return None
    feet = [i for i in (27, 28) if confidence(image_points[i]) >= min_confidence]
    if not feet:
        return None
    u = sum(image_points[i].x for i in feet)/len(feet)*calibration.width
    v = sum(image_points[i].y for i in feet)/len(feet)*calibration.height
    if not 0 <= u < calibration.width or not 0 <= v < calibration.height:
        return None
    try:
        x, z = calibration.project(u, v)
    except ValueError:
        return None
    anchor = [sum(getattr(world_points[i], axis) for i in feet)/len(feet)
              for axis in ("x", "y", "z")]
    joints = {}
    for name, index in LANDMARKS.items():
        point = world_points[index]
        local = calibration.rotate(point.x-anchor[0], -(point.y-anchor[1]),
                                   point.z-anchor[2])
        joints[name] = [x + local[0], local[1], z + local[2],
                        min(confidence(point), confidence(image_points[index]))]
    for name, left, right in (("neck", "l_shoulder", "r_shoulder"),
                              ("pelvis", "l_hip", "r_hip")):
        joints[name] = [(joints[left][i] + joints[right][i])/2 for i in range(3)] + [
            min(joints[left][3], joints[right][3])]
    if not all(math.isfinite(value) for joint in joints.values() for value in joint):
        return None
    if any(abs(value) > 10000 for joint in joints.values() for value in joint[:3]):
        return None
    return {"id": 0, "confidence": min(joints["pelvis"][3],
                                        max(joints[f"{s}_ankle"][3] for s in ("l", "r"))),
            "joints": joints}


def run(args):
    stop = threading.Event()
    if getattr(args, "stop_stdin", False):
        def read_stop():
            sys.stdin.readline()
            stop.set()
    # Optional packages are loaded here so core tests need no camera/ML install.
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    calibration = GroundCalibration.load(args.calibration)
    tracker = PersonTracker()
    capture = None
    landmarker = None
    ipc = None
    publisher = None
    pose_sent = False
    video_card = getattr(args, "video_card", False)
    # Bounded timing samples; never retain frames or skeletons across iterations.
    timing = {key: [] for key in ("capture", "inference", "mapping_tracking", "rpc", "total")}
    counts = {"frames": 0, "frames_with_pose": 0, "accepted_poses": 0,
              "rejected_poses": 0, "over_budget_frames": 0}
    if video_card:
        counts.update(card_frames=0, card_publish_drops=0)
    replay = args.video is not None
    report = {"status": "RUNNING", "image_storage": "none",
              "source": "video" if replay else "camera",
              "ipc_enabled": not args.offline,
              "latency_scope": ("capture.read through pose validation/RPC (when enabled); "
                                "excludes source buffering/render"),
              "errors": []}
    try:
        capture = (cv2.VideoCapture(str(args.video)) if replay else
                   cv2.VideoCapture(args.camera, getattr(args, "camera_backend", cv2.CAP_ANY)))
        if not capture.isOpened():
            raise RuntimeError("video could not be opened" if replay else "camera could not be opened")
        fps = capture.get(cv2.CAP_PROP_FPS) if replay else getattr(args, "camera_fps", 30.0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video must declare a positive frame rate")
        report["target_fps"] = fps
        if not replay:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, calibration.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration.height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort; backend may ignore
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.VIDEO, num_poses=args.people,
            min_pose_detection_confidence=args.confidence,
            min_pose_presence_confidence=args.confidence,
            min_tracking_confidence=args.confidence, output_segmentation_masks=video_card)
        landmarker = vision.PoseLandmarker.create_from_options(options)
        if not args.offline:
            ipc = connect(args.pipe)
            if video_card:
                if ipc.call("player.display").get("video_card") is not True:
                    raise RuntimeError("enable video_card on Player before streaming")
                publisher = VideoCardPublisher(synthetic_depth=getattr(args, "synthetic_depth", None))
        if getattr(args, "stop_stdin", False):
            print(json.dumps({"event": "ready"}), flush=True)
            threading.Thread(target=read_stop, daemon=True).start()
        start = time.perf_counter()
        previous_ms = -1
        previous_t = None
        while not stop.is_set() and not getattr(args, "stop_requested", lambda: False)():
            source_time = counts["frames"] / fps if replay else time.perf_counter() - start
            if args.seconds and source_time >= args.seconds:
                break
            if replay:
                if not args.offline:
                    stop.wait(max(0.0, start + source_time - time.perf_counter()))
                    if stop.is_set():
                        break
            else:
                wait_frame(start, counts["frames"], hz=fps)
            begin = time.perf_counter()
            ok, frame = capture.read()
            captured = time.perf_counter()
            if not ok:
                if replay and counts["frames"]:
                    break
                raise RuntimeError("camera frame unavailable; reconnect/restart bridge")
            if frame.shape[1] != calibration.width or frame.shape[0] != calibration.height:
                raise ValueError("input resolution differs from calibration; recalibrate")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_time = source_time if replay else captured - start
            timestamp = max(previous_ms + 1, int(frame_time * 1000))
            previous_ms = timestamp
            model_rgb = mask_input(rgb, cv2) if video_card else rgb
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=model_rgb), timestamp)
            inferred = time.perf_counter()
            persons = []
            card = None
            for image_points, world_points in zip(result.pose_landmarks,
                                                   result.pose_world_landmarks):
                person = convert_pose(image_points, world_points, calibration, args.confidence)
                if person is None:
                    counts["rejected_poses"] += 1
                else:
                    persons.append(person)
                    if video_card and result.segmentation_masks:
                        card = cutout(rgb, result.segmentation_masks[0].numpy_view(),
                                      image_points, person, confidence, cv2, np)
            t = timestamp/1000
            ids = tracker.update([(p["joints"]["pelvis"][0], p["joints"]["pelvis"][2])
                                  for p in persons], t)
            for person, pid in zip(persons, ids):
                person["id"] = pid
            packet = validate_pose({"t": t, "persons": persons}, previous_t)
            previous_t = t
            mapped = time.perf_counter()
            if ipc is not None:
                pose_sent = True
                ipc.call("input.set_body_poses", **packet)
            if video_card:
                counts["card_frames"] += card is not None
                if publisher is not None:
                    counts["card_publish_drops"] += not publisher.publish(card)
            end = time.perf_counter()
            values = (captured-begin, inferred-captured, mapped-inferred, end-mapped, end-begin)
            for key, value in zip(timing, values):
                timing[key].append(value*1000)
                if len(timing[key]) > 10000:
                    del timing[key][0]
            callback = getattr(args, "on_operator_frame", None)
            if callback is not None:
                callback(frame, len(persons))
            counts["frames"] += 1
            counts["frames_with_pose"] += bool(persons)
            counts["accepted_poses"] += len(persons)
            if (getattr(args, "stop_stdin", False) and persons and
                    (not video_card or card is not None) and not report.get("first_pose_ready")):
                report["first_pose_ready"] = True
                print(json.dumps({"event": "first_pose"}), flush=True)
            counts["over_budget_frames"] += end - begin > 1 / fps
            del frame, rgb, model_rgb, result, packet, persons, card
        report["status"] = "STOPPED"
    except KeyboardInterrupt:
        report["status"] = "STOPPED"
    except Exception as exc:
        report["status"] = "FAIL"
        # Error text can originate in camera/ML native code: do not persist it.
        report["errors"].append(type(exc).__name__)
    finally:
        if publisher is not None:
            try:
                publisher.close()
            except OSError as exc:
                report["errors"].append("video cleanup: " + type(exc).__name__)
                report["status"] = "FAIL"
        if ipc is not None:
            try:
                if pose_sent:
                    ipc.call("input.clear_body_poses")
            except Exception as exc:
                report["errors"].append("cleanup: " + type(exc).__name__)
                report["status"] = "FAIL"
            ipc.close()
        if landmarker is not None:
            landmarker.close()
        if capture is not None:
            capture.release()
        report.update(counts)
        report["timing_last_10000_frames"] = {key: summary_ms(value) for key, value in timing.items()}
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    return int(report["status"] == "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--video", type=Path, help="Replay a local constant-frame-rate video")
    parser.add_argument("--offline", action="store_true",
                        help="Analyze a video without connecting to Player")
    parser.add_argument("--stop-stdin", action="store_true",
                        help="Game owner protocol: ready on stdout, stop on stdin line/EOF")
    parser.add_argument("--video-card", action="store_true",
                        help="Prepare a single-person RGBA cutout; publish it when online")
    parser.add_argument("--synthetic-depth", type=float,
                        help="Vulkan test: per-pixel offset in metres along view forward; requires --video-card")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--people", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=0,
                        help="0: play to end of video or capture until Ctrl+C")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--pipe", default=PIPE)
    args = parser.parse_args()
    if args.offline and args.video is None:
        parser.error("--offline requires --video")
    if args.video_card and args.people != 1:
        parser.error("--video-card requires --people 1")
    if args.synthetic_depth is not None and (not args.video_card or
            not math.isfinite(args.synthetic_depth) or abs(args.synthetic_depth) > 20):
        parser.error("--synthetic-depth requires --video-card and a finite offset in [-20,20]")
    if not math.isfinite(args.seconds) or args.seconds < 0 or not 0 < args.confidence <= 1:
        parser.error("seconds must be finite >=0; confidence must be in (0,1]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
