from . import fire_smoke
from . import mobile_usage
from . import density_occupancy
from . import people_count
from . import smoking
from . import floating
from . import ppe
from . import falling_pose
from . import ANPR_LPR

from .registry import get_registered_kpis, list_registered_names, register_kpi

__all__ = ["get_registered_kpis", "list_registered_names", "register_kpi"]
