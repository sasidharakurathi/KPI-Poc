from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseKPI

_registry: dict[str, Type["BaseKPI"]] = {}


def register_kpi(cls: Type["BaseKPI"]) -> Type["BaseKPI"]:
    """Class decorator that registers a KPI so the pipeline discovers it."""
    _registry[cls.name] = cls
    return cls


def enabled_kpi_names() -> set[str]:
    """KPI names switched on in the KPI Management DB table
    (KPIConfiguration.enable_status) - the single source of truth for which
    KPIs run. A registered KPI with no row is treated as disabled."""
    from sqlmodel import select
    from ..db.engine import get_session_ctx
    from ..db.models.kpi_configuration import KPIConfiguration

    with get_session_ctx() as session:
        rows = session.exec(select(KPIConfiguration).where(KPIConfiguration.enable_status == True)).all()
        return {r.kpi_name for r in rows}


def get_registered_kpis() -> list["BaseKPI"]:
    """Fresh instances of every registered KPI that is enabled in the DB."""
    enabled = enabled_kpi_names()
    return [cls() for name, cls in _registry.items() if name in enabled]


def list_registered_names() -> list[str]:
    return list(_registry.keys())


def get_registry() -> dict[str, Type["BaseKPI"]]:
    """All registered KPI classes, keyed by name, regardless of enabled state."""
    return dict(_registry)
