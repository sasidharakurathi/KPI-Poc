import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


# TensorRT builder workspace (GiB) -- ultralytics defaults this high (historically 4 GiB),
# which lets the builder pick faster but more memory-hungry kernels/tactics. On a 4GB card
# that inflated runtime execution-context memory far past what a small model should need
# (e.g. cigarette.pt needed ~525MB just for its execution context). Capping it forces
# leaner tactic selection, directly shrinking each engine's runtime memory footprint.
_WORKSPACE_GIB = 1

JOBS: list[tuple[str, int, bool, int]] = [
    ("app/models/ppe.pt", 512, True, 4),
    ("app/models/yolo26s-pose.pt", 512, True, 4),        # PPEKPI's pose model
    ("app/models/fire-smoke.pt", 512, True, 4),          # batched: FireSmokeKPI runs 4 frames per call
    ("app/models/yolo26m.pt", 512, True, 4),             # mobile_usage phone + smoking person
    ("app/models/cigarette.pt", 320, True, 16),          # batch dimension is crops-per-frame, not frames
    ("app/models/ppl-count-yolo26m.pt", 640, True, 4),
    ("app/models/yolo26m-pose.pt", 640, True, 4),        # dynamic HxW (512/640 shared) + dynamic batch
    ("app/models/anpr_lpr.pt", 640, True, 4),            # currently disabled, kept in sync in case re-enabled
    ("app/models/carton-box-detection.pt", 640, True, 4),  # currently disabled, kept in sync in case re-enabled
]


def _verify_engine_profile(engine_path: Path, expected_imgsz: int, expected_max_batch: int) -> None:
    import json
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        meta_len = int.from_bytes(f.read(4), byteorder="little")
        f.read(meta_len)   # skip embedded JSON metadata
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("       FAILED to deserialize engine for verification", flush=True)
        return

    name = engine.get_tensor_name(0)   # input tensor is always index 0
    min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(name, 0)
    print(f"       profile for '{name}': min={min_shape} opt={opt_shape} max={max_shape}", flush=True)

    ok = (
        max_shape[0] >= expected_max_batch
        and min_shape[0] <= 1
        and max_shape[2] >= expected_imgsz and max_shape[3] >= expected_imgsz
        and min_shape[2] <= expected_imgsz and min_shape[3] <= expected_imgsz
    )
    print(f"       {'OK' if ok else 'WARNING — does not cover expected imgsz/batch!'}", flush=True)


def main() -> None:
    for model_path, imgsz, dynamic, max_batch in JOBS:
        engine_path = Path(model_path).with_suffix(".engine")
        if engine_path.exists():
            print(f"[skip] {engine_path} already exists")
            continue
        if not Path(model_path).exists():
            print(f"[skip] {model_path} not found")
            continue

        print(f"[export] {model_path} @ imgsz={imgsz}{f' (dynamic, max_batch={max_batch})' if dynamic else ''} ...", flush=True)
        t0 = time.perf_counter()
        try:
            model = YOLO(model_path)
            kwargs = {"batch": max_batch} if dynamic else {}
            model.export(
                format="engine", imgsz=imgsz, dynamic=dynamic, half=True, device=0,
                workspace=_WORKSPACE_GIB, **kwargs,
            )
            print(f"[done] {engine_path} ({time.perf_counter() - t0:.1f}s)", flush=True)
            _verify_engine_profile(engine_path, imgsz, max_batch)
        except Exception as e:
            print(f"[FAILED] {model_path}: {type(e).__name__}: {e}", flush=True)

    print("\nAll exports processed.")


if __name__ == "__main__":
    main()
