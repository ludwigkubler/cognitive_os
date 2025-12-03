# agents/hardware_agent.py
from __future__ import annotations

import json
import platform, psutil, GPUtil
try:
    import GPUtil  # opzionale
except ImportError:
    GPUtil = None  # type: ignore
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
    AgentRunStatus,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class HardwareAgent(Agent):
    """
    Agente non parametrico che cattura lo stato dell'hardware:
    - CPU (uso, core)
    - RAM + swap
    - Disco (per mountpoint)
    - Temperature (se disponibili)
    - GPU (se libreria GPUtil presente)
    """

    name = "hardware_agent"
    description = "Rileva lo stato corrente dell'hardware (CPU/RAM/disco/temperature/GPU)."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # Non ci interessa l'input: agente puramente "sensoriale"
        try:
            snapshot = self._gather_snapshot()
        except ImportError as e:
            msg = (
                "Per leggere lo stato hardware ho bisogno del pacchetto Python 'psutil' "
                "(e opzionalmente 'GPUtil' per la GPU).\n"
                "Installa con:\n"
                "  pip install psutil GPUtil\n"
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "error": str(e),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.03, confidence=-0.02),
                status=AgentRunStatus.FAILURE,
            )
        except Exception as e:
            msg = (
                "Ho provato a leggere lo stato hardware ma qualcosa è andato storto. "
                "Controlla i log interni per maggiori dettagli."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "error": str(e),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.04, confidence=-0.03),
                status=AgentRunStatus.FAILURE,
            )

        # Persistiamo lo snapshot in memoria globale procedurale
        try:
            memory.store_item(
                scope=MemoryScope.GLOBAL,
                type_=MemoryType.PROCEDURAL,
                key="hardware_snapshot",
                content=json.dumps(snapshot),
                metadata={
                    "created_at": snapshot["timestamp"],
                    "source_agent": self.name,
                },
            )
        except Exception:
            # non blocchiamo l'agente se la persistenza fallisce
            pass

        # Creiamo un messaggio sintetico per l'utente
        summary_text = self._build_human_summary(snapshot)

        output_payload = {
            "user_visible_message": summary_text,
            "hardware_snapshot": snapshot,
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(
            confidence=0.03,
            curiosity=0.02,
        )
        return AgentResult(output_payload=output_payload, emotion_delta=delta)

    # ---------------------------------------------------------
    # Raccolta metriche
    # ---------------------------------------------------------

    def _gather_snapshot(self) -> Dict[str, Any]:
        import psutil

        try:
            import GPUtil  # opzionale
        except ImportError:
            GPUtil = None  # type: ignore

        now = datetime.utcnow().isoformat()

        # CPU
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
        cpu_percent = psutil.cpu_percent(interval=0.5)
        per_core = psutil.cpu_percent(interval=None, percpu=True)

        load_avg: Optional[List[float]] = None
        try:
            if hasattr(psutil, "getloadavg"):
                load_avg = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load_avg = None

        # Memoria
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        # Disco
        disk_partitions = []
        for part in psutil.disk_partitions(all=False):
            # escludiamo mount "strani" (cdrom, pseudo-fs)
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except PermissionError:
                continue
            disk_partitions.append(
                {
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "percent": usage.percent,
                }
            )

        # Temperature (se disponibili)
        temps = {}
        try:
            if hasattr(psutil, "sensors_temperatures"):
                t = psutil.sensors_temperatures()
                for name, entries in t.items():
                    temps[name] = [
                        {
                            "label": e.label or "",
                            "current": e.current,
                            "high": e.high,
                            "critical": e.critical,
                        }
                        for e in entries
                    ]
        except Exception:
            temps = {}

        # GPU (se GPUtil disponibile)
        gpus_info = []
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                for g in gpus:
                    gpus_info.append(
                        {
                            "id": g.id,
                            "name": g.name,
                            "load": round(g.load * 100, 1),
                            "memory_total_mb": g.memoryTotal,
                            "memory_used_mb": g.memoryUsed,
                            "memory_free_mb": g.memoryFree,
                            "temperature": g.temperature,
                        }
                    )
            except Exception:
                gpus_info = []

        # Info OS
        system = platform.system()
        release = platform.release()
        version = platform.version()
        machine = platform.machine()

        snapshot: Dict[str, Any] = {
            "timestamp": now,
            "os": {
                "system": system,
                "release": release,
                "version": version,
                "machine": machine,
            },
            "cpu": {
                "logical_cores": logical_cores,
                "physical_cores": physical_cores,
                "percent": cpu_percent,
                "per_core_percent": per_core,
                "load_avg_1_5_15": load_avg,
            },
            "memory": {
                "total_mb": round(vm.total / (1024**2), 1),
                "used_mb": round(vm.used / (1024**2), 1),
                "available_mb": round(vm.available / (1024**2), 1),
                "percent": vm.percent,
            },
            "swap": {
                "total_mb": round(sm.total / (1024**2), 1),
                "used_mb": round(sm.used / (1024**2), 1),
                "percent": sm.percent,
            },
            "disks": disk_partitions,
            "temperatures": temps,
            "gpus": gpus_info,
        }

        return snapshot

    # ---------------------------------------------------------
    # Riassunto leggibile per l'utente
    # ---------------------------------------------------------

    def _build_human_summary(self, snapshot: Dict[str, Any]) -> str:
        cpu = snapshot["cpu"]
        mem = snapshot["memory"]
        disks = snapshot["disks"]
        temps = snapshot["temperatures"]
        gpus = snapshot["gpus"]

        lines = []

        lines.append("📟 Stato hardware attuale:")
        lines.append(
            f"- CPU: {cpu['percent']}% di utilizzo su "
            f"{cpu['logical_cores']} core logici "
            f"({cpu['physical_cores']} fisici)"
        )
        lines.append(
            f"- RAM: {mem['used_mb']:.0f} / {mem['total_mb']:.0f} MB "
            f"({mem['percent']}% in uso)"
        )

        if disks:
            # prendiamo solo il primo disco per il testo breve
            main_disk = disks[0]
            lines.append(
                f"- Disco principale ({main_disk['mountpoint']}): "
                f"{main_disk['used_gb']:.1f} / {main_disk['total_gb']:.1f} GB "
                f"({main_disk['percent']}% in uso)"
            )
        else:
            lines.append("- Disco: nessuna partizione leggibile trovata.")

        # Temperature sintetiche
        if temps:
            # cerchiamo cpu o package
            cpu_temp_line = None
            for name, entries in temps.items():
                for entry in entries:
                    label = entry.get("label") or name
                    if "cpu" in label.lower() or "package" in label.lower():
                        cpu_temp_line = f"- Temperatura CPU: {entry['current']}°C (sensore {label})"
                        break
                if cpu_temp_line:
                    break
            if cpu_temp_line:
                lines.append(cpu_temp_line)
            else:
                lines.append("- Temperature: sensori disponibili, ma nessun label CPU riconosciuto.")
        else:
            lines.append("- Temperature: nessun sensore disponibile o permessi insufficienti.")

        # GPU sintetica
        if gpus:
            g = gpus[0]
            lines.append(
                f"- GPU: {g['name']} al {g['load']}% di utilizzo, "
                f"{g['memory_used_mb']:.0f}/{g['memory_total_mb']:.0f} MB VRAM, "
                f"temperatura {g['temperature']}°C"
            )

        lines.append("\nHo salvato uno snapshot dettagliato nella memoria interna (chiave: hardware_snapshot).")

        return "\n".join(lines)
