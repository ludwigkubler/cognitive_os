from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import List

from .agents_base import Agent, AgentRegistry


def load_agents_from_packages(
    registry: AgentRegistry,
    package_names: List[str],
) -> None:
    """
    Scansiona i package indicati (es. ["agents", "r_agents"]),
    importa tutti i moduli e registra tutte le sottoclassi concrete di Agent.
    """
    for pkg_name in package_names:
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError as exc:  # noqa: BLE001
            print(f"[AGENT_LOADER] Impossibile importare package '{pkg_name}': {exc}")
            continue

        if not hasattr(pkg, "__path__"):
            # non è un package con sottomoduli
            continue

        for finder, mod_name, ispkg in pkgutil.iter_modules(pkg.__path__):
            full_name = f"{pkg_name}.{mod_name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[AGENT_LOADER] Errore importando modulo '{full_name}': {exc}")
                continue

            for _, obj in module.__dict__.items():
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Agent)
                    and obj is not Agent
                ):
                    try:
                        instance = obj()
                        registry.register(instance)
                        print(f"[AGENT_LOADER] Registrato agent '{instance.name}' da {full_name}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[AGENT_LOADER] Errore istanziando agent in '{full_name}': {exc}")
