"""Multi-object tracker wrapper — one stateful instance PER camera (build-guide Step 4).

supervision 0.29 ships ByteTrack natively, which is the fast default and all that
occupancy + dwell need. BoT-SORT (appearance ReID, steadier IDs under heavy occlusion)
is NOT part of supervision; it lives in the Ultralytics-native `model.track(...,
tracker="botsort.yaml")` path. Rather than silently pretend, we log and fall back to
ByteTrack when `type: botsort` is requested, so dwell IDs stay well-defined. Wiring the
Ultralytics-native BoT-SORT path is a clean Phase-7 extension.
"""
import supervision as sv


class Tracker:
    def __init__(self, tcfg, frame_rate=30):
        self.type = tcfg.get("type", "bytetrack").lower()
        if self.type == "botsort":
            print("[tracker] BoT-SORT not available via supervision 0.29 — falling back "
                  "to ByteTrack. (See tracker.py docstring for the ReID path.)")
        self.impl = sv.ByteTrack(
            track_activation_threshold=float(tcfg.get("track_activation_threshold", 0.25)),
            lost_track_buffer=int(tcfg.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(tcfg.get("minimum_matching_threshold", 0.80)),
            frame_rate=int(frame_rate) if frame_rate and frame_rate > 0 else 30,
        )

    def update(self, detections):
        """detections: sv.Detections -> same, now carrying .tracker_id."""
        return self.impl.update_with_detections(detections)
