import cv2
import numpy as np

from .kpis.base import KPIResult

_FONT           = cv2.FONT_HERSHEY_COMPLEX
_BOX_THICKNESS  = 2
_LABEL_SCALE    = 0.38
_LABEL_THICK    = 1
_SHADOW_COLOR   = (0, 0, 0)


def _draw_detection(frame: np.ndarray, det, fallback_color: tuple) -> None:
    color = det.color if det.color is not None else fallback_color

    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, _BOX_THICKNESS)

    label = (
        f"{det.label}  {det.confidence:.0%}"
        if det.confidence < 1.0
        else det.label
    )

    (tw, th), baseline = cv2.getTextSize(label, _FONT, _LABEL_SCALE, _LABEL_THICK)
    pad    = 5
    bg_y1  = max(0, det.y1 - th - pad * 2 - baseline)
    bg_y2  = det.y1
    bg_x2  = det.x1 + tw + pad * 2

    cv2.rectangle(frame, (det.x1, bg_y1), (bg_x2, bg_y2), color, -1)

    text_y = bg_y2 - baseline - 2
    # Text shadow then white text
    cv2.putText(frame, label, (det.x1 + pad + 1, text_y + 1), _FONT, _LABEL_SCALE, _SHADOW_COLOR, _LABEL_THICK + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (det.x1 + pad, text_y),         _FONT, _LABEL_SCALE, (255, 255, 255), _LABEL_THICK, cv2.LINE_AA)


def _draw_status_panel(
    frame: np.ndarray,
    kpi_results: dict[str, KPIResult],
    frame_idx: int,
) -> None:
    sections: list[tuple[str, tuple, list[str]]] = []
    for result in kpi_results.values():
        if frame_idx < len(result.frame_annotations):
            ann = result.frame_annotations[frame_idx]
            if ann.status_lines:
                sections.append((result.display_name, result.color, ann.status_lines))

    if not sections:
        return

    pad       = 8
    title_fs  = 0.44
    line_fs   = 0.38
    title_th  = 1
    line_th   = 1
    line_h    = 20
    accent_w  = 4

    # Measure panel width
    max_w = 0
    for title, _, lines in sections:
        max_w = max(max_w, cv2.getTextSize(title, _FONT, title_fs, title_th)[0][0])
        for line in lines:
            max_w = max(max_w, cv2.getTextSize(line, _FONT, line_fs, line_th)[0][0])

    total_lines = sum(1 + len(lines) for _, _, lines in sections)
    panel_w     = max_w + pad * 2 + accent_w + 6
    panel_h     = total_lines * line_h + pad * 2

    x0, y0 = 12, 12

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    cy = y0 + pad + line_h - 4
    for title, color, lines in sections:
        section_h = (1 + len(lines)) * line_h
        cv2.rectangle(frame, (x0, cy - line_h + 2), (x0 + accent_w, cy - line_h + 2 + section_h), color, -1)

        tx = x0 + accent_w + 8

        # Title: thin colored outline then white fill
        cv2.putText(frame, title, (tx, cy), _FONT, title_fs, color,           title_th , cv2.LINE_AA)
        cv2.putText(frame, title, (tx, cy), _FONT, title_fs, (255, 255, 255), title_th,     cv2.LINE_AA)
        cy += line_h

        for line in lines:
            cv2.putText(frame, line, (tx, cy), _FONT, line_fs, (210, 210, 210), line_th, cv2.LINE_AA)
            cy += line_h


def compose_video(
    source_path: str,
    kpi_results: dict[str, KPIResult],
    output_path: str,
) -> None:
    cap    = cv2.VideoCapture(source_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
