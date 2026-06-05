from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseKPI

_registry: dict[str, Type["BaseKPI"]] = {}


def register_kpi(cls: Type["BaseKPI"]) -> Type["BaseKPI"]:
    """
    Class decorator that registers a KPI so the pipeline discovers it.

    Usage:
        @register_kpi
        class MyKPI(BaseKPI):
            name = "my_kpi"
            ...
    """
    _registry[cls.name] = cls
    return cls


def get_registered_kpis() -> list["BaseKPI"]:
    """
    Return fresh instances of every registered KPI that has enabled=true
    (or no 'enabled' key) in config.json.
    """
    from ..config_loader import get_kpi_param
    instances = []
    for cls in _registry.values():
        enabled = get_kpi_param(cls.__name__, "enabled", True)
        if enabled:
            instances.append(cls())
    return instances


def list_registered_names() -> list[str]:
    return list(_registry.keys())
