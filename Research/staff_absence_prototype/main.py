"""Staff Absence prototype runner.

Pipeline (mirrors occupancy_prototype/main.py's shape, which mirrors pose_prototype's):
  ingest -> detect (body/head mode) -> track (ByteTrack) -> per-zone Staff Absence
         -> overlay -> debounced events (+ evidence snapshot).

Guard Presence (KPI #19) is currently a placeholder: any tracked person confirmed inside the
zone counts as "present" (see guard_presence.py). Staff Absence times how long the zone stays
continuously empty during the guard's scheduled shift (see absence.py) and alerts past a
configurable threshold.

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
from absence import StaffAbsenceZone
from event_layer import EventBus

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pill(img, text, x, y, fg, bg, fs=0.5, ft=1, pad=4):
    (tw, th), bl = cv2.getTextSize(text, _FONT, fs, ft)
    cv2.rectangle(img, (x, y), (x + tw + pad * 2, y + th + bl + pad * 2), bg, cv2.FILLED)
    cv2.putText(img, text, (x + pad, y + th + pad), _FONT, fs, fg, ft, cv2.LINE_AA)
    return y + th + bl + pad * 2


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
    acfg = cfg.get("alerts", {})
    alerts_dir = acfg.get("dir", "alerts")
    if not os.path.isabs(alerts_dir):
        alerts_dir = os.path.join(here, alerts_dir)
    save_snapshot = acfg.get("save_snapshot", True)
    if save_snapshot:
        os.makedirs(alerts_dir, exist_ok=True)

    source = int(args.source) if str(args.source).isdigit() else args.source
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
    zones = [StaffAbsenceZone(z, persist_frames=cfg.get("occupancy_persist_frames", 3),
                              miss_grace=ecfg.get("gap_tol", 5))
             for z in cfg["zones"]]
    bus = EventBus(ecfg["persist_frames"], ecfg["cooldown_secs"], ecfg.get("gap_tol", 2))
    print(f"[init] camera={cam} zones={[z.name for z in zones]} "
          f"src_fps={src_fps:.1f} stride={stride} eff_fps={eff_fps:.1f}")

    writer = None
    WIN = "staff absence prototype (ESC to quit)"
    if args.view:
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

        # draw all person boxes + track id
        if dcfg.get("boxes", True) and det.tracker_id is not None:
            for box, tid in zip(det.xyxy, det.tracker_id):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
                if dcfg.get("track_labels", True) and tid is not None:
                    _pill(annotated, f"#{int(tid)}", x1, max(0, y1 - 20),
                          (255, 255, 255), (40, 40, 40), fs=0.45)

        # per-zone staff absence
        panel_y = 8
        for zone in zones:
            res = zone.update(det, ts)

            # zone polygon: green = present, orange = empty (off-shift or within grace),
            # red = absence alert firing
            if res["absence_alert"]:
                col = (0, 0, 220)
            elif res["present"]:
                col = (0, 200, 60)
            else:
                col = (0, 160, 255)
            if dcfg.get("zone_fill", True):
                ov = annotated.copy()
                cv2.fillPoly(ov, [zone.polygon], col)
                cv2.addWeighted(ov, 0.18, annotated, 0.82, 0, annotated)
            cv2.polylines(annotated, [zone.polygon], True, col, 2)

            # top-left status panel
            if not res["schedule_active"]:
                txt = f"{zone.name}: OFF-SHIFT"
                bg = (90, 90, 90)
            elif res["present"]:
                txt = f"{zone.name}: PRESENT"
                bg = (0, 140, 40)
            else:
                txt = f"{zone.name}: ABSENT {res['absent_secs']:4.1f}s"
                bg = (0, 0, 220) if res["absence_alert"] else (0, 110, 200)
            if dcfg.get("presence_label", True):
                panel_y = _pill(annotated, txt, 8, panel_y, (255, 255, 255), bg, fs=0.6) + 3

            # events (debounced + cooldown)
            if bus.submit((zone.name, "absence"), res["absence_alert"], ts):
                print(f"[EVENT] t={ts:7.1f}s  zone={zone.name:<10} ABSENCE   "
                      f"absent={res['absent_secs']:.1f}s")
                if save_snapshot:
                    fname = f"{zone.name}_{ts:.1f}s.jpg".replace(":", "-")
                    cv2.imwrite(os.path.join(alerts_dir, fname), annotated)

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

        if args.view:
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
    if args.view:
        cv2.destroyAllWindows()
    print(f"[done] processed {proc} frames  avg {fps_ema:.1f} FPS")


if __name__ == "__main__":
    main()
