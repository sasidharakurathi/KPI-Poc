import cv2
import numpy as np

from .kpis.base import KPIResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOX_THICKNESS = 2
_LABEL_FONT_SCALE = 0.5
_LABEL_THICKNESS = 1


def _draw_detection(frame: np.ndarray, det, fallback_color: tuple) -> None:
    color = det.color if det.color is not None else fallback_color
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, _BOX_THICKNESS)

    label = (
        f"{det.label} {det.confidence:.2f}"
        if det.confidence < 1.0
        else det.label
    )
    (tw, th), _ = cv2.getTextSize(label, _FONT, _LABEL_FONT_SCALE, _LABEL_THICKNESS)
    bg_y1 = max(0, det.y1 - th - 8)
    cv2.rectangle(frame, (det.x1, bg_y1), (det.x1 + tw + 4, det.y1), color, -1)
    cv2.putText(
        frame, label,
        (det.x1 + 2, det.y1 - 4 if det.y1 > 14 else det.y1 + th + 4),
        _FONT, _LABEL_FONT_SCALE, (255, 255, 255), _LABEL_THICKNESS,
    )


def _draw_status_panel(
    frame: np.ndarray,
    kpi_results: dict[str, KPIResult],
    frame_idx: int,
) -> None:
    """Top-left semi-transparent panel listing every KPI's status lines."""
    sections: list[tuple[str, tuple, list[str]]] = []
    for result in kpi_results.values():
        if frame_idx < len(result.frame_annotations):
            ann = result.frame_annotations[frame_idx]
            if ann.status_lines:
                sections.append((result.display_name, result.color, ann.status_lines))

    if not sections:
        return

    pad = 6
    line_h = 20
    fs = 0.45
    th = 1

    # Measure panel width
    max_w = 0
    for title, _, lines in sections:
        max_w = max(max_w, cv2.getTextSize(title, _FONT, fs + 0.05, th + 1)[0][0])
        for line in lines:
            max_w = max(max_w, cv2.getTextSize(line, _FONT, fs, th)[0][0])

    total_lines = sum(1 + len(lines) for _, _, lines in sections)
    panel_w = max_w + pad * 2
    panel_h = total_lines * line_h + pad * 2

    x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cy = y0 + pad + line_h
    for title, color, lines in sections:
        cv2.putText(frame, title, (x0 + pad, cy), _FONT, fs + 0.05, color, th + 1)
        cy += line_h
        for line in lines:
            cv2.putText(frame, line, (x0 + pad + 8, cy), _FONT, fs, (210, 210, 210), th)
            cy += line_h


def compose_video(
    source_path: str,
    kpi_results: dict[str, KPIResult],
    output_path: str,
) -> None:
    """
    Read source_path frame-by-frame, overlay all KPI annotations,
    and write the result to output_path as an mp4.
    """
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for result in kpi_results.values():
            if frame_idx < len(result.frame_annotations):
                for det in result.frame_annotations[frame_idx].detections:
                    _draw_detection(frame, det, result.color)

        _draw_status_panel(frame, kpi_results, frame_idx)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
