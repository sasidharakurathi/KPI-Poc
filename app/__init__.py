import logging
import os

import ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        try:
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        val = line.split("=", 1)[1].strip()
                        break
        except OSError:
            pass
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


_max_workers = max(1, _env_int("MAX_WORKERS", 4))
_threads_per_worker = max(1, (os.cpu_count() or 4) // _max_workers)

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(_threads_per_worker))
