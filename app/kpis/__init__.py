from . import fire_smoke
from . import mobile_usage
from . import density_occupancy
from . import people_count
from . import smoking
from . import floating
from . import ppe
from . import falling_pose
from . import ANPR_LPR
from . import carton_box_detection
from . import vehicle_detection_speed
from . import Uniform_detection


from .registry import get_registered_kpis, get_registry, list_registered_names, register_kpi

__all__ = ["get_registered_kpis", "get_registry", "list_registered_names", "register_kpi"]
