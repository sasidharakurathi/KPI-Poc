import logging
import os

# Silence ultralytics' own logger (e.g. the "'half' is deprecated..." warning
# fired on every single predict()/track() call across all 8 KPIs — floods
# the systemd journal without indicating an actual problem). Real errors
# still surface via exceptions, not this logger.
#
# ultralytics installs its own handler + INFO level the first time the
# package is imported, which would clobber a level set beforehand — so
# import it here first, then override.
import ultralytics  # noqa: F401
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
