"""Click four measured floor markers on live preview. Saves coordinates only."""
import argparse
import json
from pathlib import Path

try:
    from .calibration import GroundCalibration
except ImportError:
    from calibration import GroundCalibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    parser.add_argument("--room-xz", type=float, nargs=8, required=True,
                        help="X Z of four floor markers, in click order, metres")
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--roll", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    capture = cv2.VideoCapture(args.camera)
    pixels = []
    title = "Floor calibration: click 4 markers; Enter saves; R resets; Esc cancels"

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(pixels) < 4:
            pixels.append([x, y])

    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.size[1])
        if not capture.isOpened():
            raise RuntimeError("camera could not be opened")
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, click)
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame unavailable")
            size = [frame.shape[1], frame.shape[0]]
            for i, (x, y) in enumerate(pixels):
                cv2.circle(frame, (x, y), 6, (0, 255, 0), 2)
                cv2.putText(frame, str(i+1), (x+8, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            cv2.imshow(title, frame)
            key = cv2.waitKey(20) & 255
            del frame
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return 1
            if key in (ord("r"), ord("R")):
                pixels.clear()
            if key == 13 and len(pixels) == 4:
                config = {"image_size": size, "camera_yaw_degrees": args.yaw,
                          "camera_pitch_degrees": args.pitch, "camera_roll_degrees": args.roll,
                          "points": [{"pixel": p, "room_xz": args.room_xz[2*i:2*i+2]}
                                     for i, p in enumerate(pixels)]}
                GroundCalibration(config)  # reject singular correspondence before writing
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
