"""Occupancy + Dwell prototype runner.

Pipeline (mirrors pose_prototype/main.py's shape):
  ingest -> detect (body/head mode) -> track (ByteTrack) -> per-zone occupancy+dwell
         -> overlay -> debounced events.

Everything downstream of the detector works in ORIGINAL-frame pixel coords, so zones
drawn with zone_drawer.py line up regardless of detector `imgsz`.

Examples:
  python main.py --source test_videos/people_walking.avi
  python main.py --source 0                                   # webcam
  python main.py --source clip.mp4 --save out.mp4 --no-view
"""
import argparse, os, time
import cv2
import numpy as np
import yaml

from detector import PeopleDetector
from tracker import Tracker
from occupancy import OccupancyZone
from event_layer import EventBus

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pill(img, text, x, y, fg, bg, fs=0.5, ft=1, pad=4):
    (tw, th), bl = cv2.getTextSize(text, _FONT, fs, ft)
    cv2.rectangle(img, (x, y), (x + tw + pad * 2, y + th + bl + pad * 2), bg, cv2.FILLED)
    cv2.putText(img, text, (x + pad, y + th + pad), _FONT, fs, fg, ft, cv2.LINE_AA)
    return y + th + bl + pad * 2


def gui_available():
    """Some OpenCV builds (opencv-python-headless) have no HighGUI support."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def _safe_destroy_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="test_videos/people_walking.avi")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--save", default=None)
    p.add_argument("--view", dest="view", action="store_true", default=True)
    p.add_argument("--no-view", dest="view", action="store_false")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--disp-max", type=int, default=1100)
    return p.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(here, args.config)
    cfg = yaml.safe_load(open(cfg_path))
    cam = cfg.get("camera_id", "cam")
    dcfg = cfg["display"]

    source = int(args.source) if str(args.source).isdigit() else args.source
    # resolve relative video paths against the prototype dir
    if isinstance(source, str) and not os.path.isabs(source) and not os.path.exists(source):
        alt = os.path.join(here, source)
        if os.path.exists(alt):
            source = alt
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    stride = max(1, int(cfg.get("frame_stride", 1)))
    eff_fps = (src_fps / stride) if src_fps > 0 else 30.0

    detector = PeopleDetector(cfg["detector"])
    tracker = Tracker(cfg["tracker"], frame_rate=eff_fps)
    ecfg = cfg["event"]
    zones = [OccupancyZone(z, persist_frames=cfg.get("occupancy_persist_frames", 3),
                           miss_grace=ecfg.get("gap_tol", 5))
             for z in cfg["zones"]]
    bus = EventBus(ecfg["persist_frames"], ecfg["cooldown_secs"], ecfg.get("gap_tol", 2))
    print(f"[init] camera={cam} zones={[z.name for z in zones]} "
          f"src_fps={src_fps:.1f} stride={stride} eff_fps={eff_fps:.1f}")

    writer = None
    WIN = "occupancy prototype (ESC to quit)"
    view = args.view and gui_available()
    if args.view and not view:
        print("[view] OpenCV has no GUI support (headless build) — running without a window. "
              "Use --save to write an annotated video.")
    if view:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    frame_idx = 0
    proc = 0
    t_last = time.time()
    fps_ema = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx - 1) % stride != 0:      # frame-stride cadence
            continue
        proc += 1

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if isinstance(source, str) and pos_ms > 0:
            ts = pos_ms / 1000.0
        elif isinstance(source, str) and src_fps > 0:
            ts = frame_idx / src_fps
        else:
            ts = time.time()

        frame = detector.preprocess(frame)          # fisheye dewarp if enabled
        det = detector.detect(frame)
        det = tracker.update(det)

        annotated = frame.copy()
        # map track id -> box for dwell labels
        id2box = {}
        if det.tracker_id is not None:
            for box, tid in zip(det.xyxy, det.tracker_id):
                if tid is not None:
                    id2box[int(tid)] = box

        # draw all person/head boxes + track id
        if dcfg.get("boxes", True) and det.tracker_id is not None:
            for box, tid in zip(det.xyxy, det.tracker_id):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
                if dcfg.get("track_labels", True) and tid is not None:
                    _pill(annotated, f"#{int(tid)}", x1, max(0, y1 - 20),
                          (255, 255, 255), (40, 40, 40), fs=0.45)

        # per-zone occupancy + dwell
        panel_y = 8
        for zone in zones:
            res = zone.update(det, ts)

            # zone polygon (fill + outline); red when overcrowded
            col = (0, 80, 220) if res["overcrowd"] else (0, 200, 255)
            if dcfg.get("zone_fill", True):
                ov = annotated.copy()
                cv2.fillPoly(ov, [zone.polygon], col)
                cv2.addWeighted(ov, 0.18, annotated, 0.82, 0, annotated)
            cv2.polylines(annotated, [zone.polygon], True, col, 2)

            # dwell labels on each confirmed-inside person
            alert_ids = {tid for tid, _ in res["dwell_alerts"]}
            if dcfg.get("dwell_labels", True):
                for tid, dwell in res["people"]:
                    if tid in id2box:
                        x1, y1, x2, y2 = map(int, id2box[tid])
                        hot = tid in alert_ids
                        _pill(annotated, f"{dwell:4.1f}s", x1, y2 + 2,
                              (255, 255, 255), (0, 0, 210) if hot else (120, 90, 0),
                              fs=0.45)

            # top-left status panel
            txt = f"{zone.name}: occ {res['occupancy']}"
            if res["overcrowd"]:
                txt += f" (>{zone.occupancy_alert}!)"
            panel_y = _pill(annotated, txt, 8, panel_y, (255, 255, 255),
                            (0, 80, 220) if res["overcrowd"] else (50, 50, 50), fs=0.6) + 3

            # events (debounced + cooldown)
            for tid, dwell in res["dwell_alerts"]:
                if bus.submit((zone.name, "dwell", tid), True, ts):
                    print(f"[EVENT] t={ts:7.1f}s  zone={zone.name:<8} DWELL     "
                          f"track={tid} dwell={dwell:.1f}s")
            if bus.submit((zone.name, "overcrowd"), res["overcrowd"], ts):
                print(f"[EVENT] t={ts:7.1f}s  zone={zone.name:<8} OVERCROWD occ={res['occupancy']}")

        # fps meter
        now = time.time()
        inst = 1.0 / max(now - t_last, 1e-6)
        t_last = now
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        rt = f" ({fps_ema / src_fps:.1f}x rt)" if src_fps > 0 else ""
        _pill(annotated, f"{fps_ema:5.1f} FPS{rt}", 8, annotated.shape[0] - 30,
              (0, 0, 0), (0, 220, 220), fs=0.55)

        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         eff_fps if eff_fps > 0 else 25.0, (w, h))
            writer.write(annotated)

        if view:
            disp = annotated
            h, w = disp.shape[:2]
            s = min(1.0, args.disp_max / max(w, h))
            if s < 1.0:
                disp = cv2.resize(disp, (int(w * s), int(h * s)))
            cv2.imshow(WIN, disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        if args.max_frames and proc >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[done] saved -> {args.save}")
    if view:
        _safe_destroy_windows()
    print(f"[done] processed {proc} frames  avg {fps_ema:.1f} FPS")


if __name__ == "__main__":
    main()
