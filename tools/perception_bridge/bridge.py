"""Webcam -> MediaPipe multi-pose -> room joints -> external Player IPC.

Frames stay in memory. No preview, image encoding, recording or landmark logs.
Only aggregate timing/error counts may be written to --metrics.
"""
import argparse
import json
import math
from pathlib import Path
import time

try:
    from .calibration import GroundCalibration
    from .schema import validate_pose
    from .tracking import PersonTracker
    from .transport import PIPE, connect, summary_ms, wait_frame
except ImportError:
    from calibration import GroundCalibration
    from schema import validate_pose
    from tracking import PersonTracker
    from transport import PIPE, connect, summary_ms, wait_frame

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
    # Optional packages are loaded here so core tests need no camera/ML install.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    calibration = GroundCalibration.load(args.calibration)
    tracker = PersonTracker()
    capture = None
    landmarker = None
    ipc = None
    # Bounded timing samples; never retain frames or skeletons across iterations.
    timing = {key: [] for key in ("capture", "inference", "mapping_tracking", "rpc", "total")}
    counts = {"frames": 0, "rejected_poses": 0, "over_budget_frames": 0}
    report = {"status": "RUNNING", "image_storage": "none",
              "latency_scope": "capture.read to pose RPC acknowledgement; excludes sensor buffering/render",
              "errors": []}
    try:
        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            raise RuntimeError("camera could not be opened")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, calibration.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration.height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort; backend may ignore
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.VIDEO, num_poses=args.people,
            min_pose_detection_confidence=args.confidence,
            min_pose_presence_confidence=args.confidence,
            min_tracking_confidence=args.confidence, output_segmentation_masks=False)
        landmarker = vision.PoseLandmarker.create_from_options(options)
        ipc = connect(args.pipe)
        start = time.perf_counter()
        previous_ms = -1
        previous_t = None
        while args.seconds == 0 or time.perf_counter()-start < args.seconds:
            wait_frame(start, counts["frames"])
            begin = time.perf_counter()
            ok, frame = capture.read()
            captured = time.perf_counter()
            if not ok:
                raise RuntimeError("camera frame unavailable; reconnect/restart bridge")
            if frame.shape[1] != calibration.width or frame.shape[0] != calibration.height:
                raise ValueError("camera resolution differs from calibration; recalibrate")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = max(previous_ms + 1, int((captured-start)*1000))
            previous_ms = timestamp
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
            inferred = time.perf_counter()
            persons = []
            for image_points, world_points in zip(result.pose_landmarks,
                                                   result.pose_world_landmarks):
                person = convert_pose(image_points, world_points, calibration, args.confidence)
                if person is None:
                    counts["rejected_poses"] += 1
                else:
                    persons.append(person)
            t = timestamp/1000
            ids = tracker.update([(p["joints"]["pelvis"][0], p["joints"]["pelvis"][2])
                                  for p in persons], t)
            for person, pid in zip(persons, ids):
                person["id"] = pid
            packet = validate_pose({"t": t, "persons": persons}, previous_t)
            previous_t = t
            mapped = time.perf_counter()
            ipc.call("input.set_body_poses", **packet)
            end = time.perf_counter()
            values = (captured-begin, inferred-captured, mapped-inferred, end-mapped, end-begin)
            for key, value in zip(timing, values):
                timing[key].append(value*1000)
                if len(timing[key]) > 10000:
                    del timing[key][0]
            counts["frames"] += 1
            counts["over_budget_frames"] += end-begin > 1/30
            del frame, rgb, result, packet, persons
        report["status"] = "STOPPED"
    except KeyboardInterrupt:
        report["status"] = "STOPPED"
    except Exception as exc:
        report["status"] = "FAIL"
        # Error text can originate in camera/ML native code: do not persist it.
        report["errors"].append(type(exc).__name__)
    finally:
        if ipc is not None:
            try:
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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--people", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=0, help="0: run until Ctrl+C")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--pipe", default=PIPE)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0 or not 0 < args.confidence <= 1:
        parser.error("seconds must be finite >=0; confidence must be in (0,1]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
