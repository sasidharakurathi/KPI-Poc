"""Click a polygon on a grabbed frame, print the points for config.yaml.

Zones are in ORIGINAL-frame pixel coords (Ultralytics rescales detections back to the
original frame, so that is the coordinate system occupancy runs in — independent of
detector `imgsz`). If fisheye dewarp is enabled for the camera, grab the frame WITH
--dewarp so the polygon matches the dewarped geometry the pipeline sees.

Usage:
  python zone_drawer.py --source test_videos/people_walking.avi
  python zone_drawer.py --source frame.jpg --frame 0
Controls: left-click add point · u undo · c clear · s print+save · Esc quit.
"""
import argparse, os
import cv2
import numpy as np
import yaml


def grab_frame(source, frame_no):
    if os.path.splitext(str(source))[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"could not read image {source}")
        return img
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")
    if frame_no > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not grab frame")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0, help="frame index to grab from a video")
    ap.add_argument("--dewarp", default=None, help="camera config.yaml to apply fisheye dewarp")
    ap.add_argument("--disp-max", type=int, default=1100)
    args = ap.parse_args()

    frame = grab_frame(args.source, args.frame)

    if args.dewarp:
        cfg = yaml.safe_load(open(args.dewarp))
        fcfg = cfg.get("detector", {}).get("fisheye", {})
        if fcfg.get("enabled"):
            from detector import FisheyeDewarper
            frame = FisheyeDewarper(fcfg)(frame)

    H, W = frame.shape[:2]
    scale = min(1.0, args.disp_max / max(W, H))
    disp0 = cv2.resize(frame, (int(W * scale), int(H * scale))) if scale < 1.0 else frame.copy()
    pts = []   # original-frame coords

    def redraw():
        d = disp0.copy()
        for i, (x, y) in enumerate(pts):
            dp = (int(x * scale), int(y * scale))
            cv2.circle(d, dp, 4, (0, 255, 0), -1)
            if i > 0:
                pp = (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale))
                cv2.line(d, pp, dp, (0, 255, 0), 2)
        if len(pts) >= 3:
            poly = np.array([[int(x * scale), int(y * scale)] for x, y in pts])
            cv2.polylines(d, [poly], True, (0, 200, 255), 2)
        cv2.putText(d, f"{len(pts)} pts  [click add | u undo | c clear | s save | Esc]",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("zone_drawer", d)

    def on_mouse(e, x, y, flags, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x / scale), int(y / scale)])
            redraw()

    cv2.namedWindow("zone_drawer")
    cv2.setMouseCallback("zone_drawer", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k in (ord("u"), ord("U")) and pts:
            pts.pop(); redraw()
        elif k in (ord("c"), ord("C")):
            pts.clear(); redraw()
        elif k in (ord("s"), ord("S")):
            print("\npolygon:", pts)
            print("paste into config.yaml under the zone's `polygon:` key\n")
    cv2.destroyAllWindows()
    print("FINAL polygon:", pts)


if __name__ == "__main__":
    main()
