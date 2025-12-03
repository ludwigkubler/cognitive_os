# curator_agent.py — versione 2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider

LIFECYCLE_STATES = ("draft", "test", "active", "deprecated")


class CuratorAgent(Agent):
    """
    CuratorAgent 2.0 — Governance formale del ciclo di vita degli agent.

    Funzioni principali:
    - legge suggerimenti del CriticAgent (promote/deprecate)
    - legge metriche di DiagnosticsAgent (success/failure)
    - legge alert di SecurityReview
    - applica policy di promozione / deprecazione
    - registra genealogia e versioni
    - mantiene la coerenza del registro agent_definitions
    """

    name = "curator_agent"
    description = "Governance formale del ciclo di vita degli agent (draft/test/active/deprecated), con versioning e genealogia."

    # -----------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------

    def _get_agent_state(self, d: Dict[str, Any]) -> str:
        """Restituisce lo stato corrente dell'agente (default: draft)."""
        st = d.get("lifecycle_state") or d.get("state")
        if st not in LIFECYCLE_STATES:
            return "draft"
        return st

    def _save(self, memory: MemoryEngine, d: Dict[str, Any]):
        """Wrapper safe per save_agent_definition."""
        try:
            memory.save_agent_definition(d)
        except Exception:
            pass

    def _load_suggestions(self, memory: MemoryEngine) -> List[Dict[str, Any]]:
        """
        Carica eventuali suggerimenti del CriticAgent:
        scope=GLOBAL / PROJECT
        type=PROCEDURAL
        key="critic_suggestion"
        """
        try:
            return memory.find_items_by_key("critic_suggestion")
        except Exception:
            return []

    def _load_security_alerts(self, memory: MemoryEngine) -> List[Dict[str, Any]]:
        try:
            return memory.find_items_by_key("security_alert")
        except Exception:
            return []

    def _load_diagnostics(self, memory: MemoryEngine) -> Dict[str, Dict[str, float]]:
        """
        Recupera metriche diagnostiche:
        {
          "agent_name": { "success_rate": 0.9, "failure_rate": 0.1, "avg_time": 1.2 }
        }
        """
        try:
            items = memory.find_items_by_key("agent_metrics")
        except Exception:
            return {}

        metrics = {}
        for item in items:
            try:
                obj = item.as_dict()
                ag = obj.get("agent_name")
                if ag:
                    metrics[ag] = obj
            except Exception:
                continue
        return metrics

    def _append_genealogy(
        self,
        memory: MemoryEngine,
        agent_name: str,
        parent: Optional[str],
        version: Optional[str],
        reason: str
    ):
        """Registra genealogia versioni."""
        record = {
            "agent": agent_name,
            "parent": parent,
            "version": version,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            memory.store_item(
                scope=MemoryScope.GLOBAL,
                type_=MemoryType.PROCEDURAL,
                key="genealogy_record",
                content=record
            )
        except Exception:
            pass

    # -----------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState
    ) -> AgentResult:

        # 1. Carica tutte le definizioni agent
        if not hasattr(memory, "list_agent_definitions"):
            return AgentResult({"user_visible_message": "Agent definitions non disponibili."})

        defs = memory.list_agent_definitions()
        if not defs:
            return AgentResult({"user_visible_message": "Nessun agent registrato."})

        # 2. Carica suggerimenti CriticAgent
        suggestions = self._load_suggestions(memory)

        # 3. Carica metriche diagnostiche
        metrics = self._load_diagnostics(memory)

        # 4. Carica alert sicurezza
        alerts = self._load_security_alerts(memory)

        # 5. Policy decisioni
        promotion_applied = []
        demotion_applied = []

        for d in defs:
            name = d.get("name")
            state = self._get_agent_state(d)

            # ---- (A) Sicurezza ha priorità: deprecazione immediata ----
            for al in alerts:
                if al.content.get("agent") == name:
                    d["lifecycle_state"] = "deprecated"
                    self._save(memory, d)
                    demotion_applied.append((name, "security_violation"))
                    self._append_genealogy(memory, name, parent=name, version=d.get("version"), reason="security_violation")
                    continue

            # ---- (B) CriticAgent può suggerire promozioni / deprecazioni ----
            for sug in suggestions:
                c = sug.content
                if c.get("agent") != name:
                    continue

                action = c.get("action")
                reason = c.get("reason", "")

                # Degradazione
                if action == "deprecate":
                    d["lifecycle_state"] = "deprecated"
                    self._save(memory, d)
                    demotion_applied.append((name, reason))
                    self._append_genealogy(memory, name, parent=name, version=d.get("version"), reason=reason)
                    continue

                # Promozione
                if action == "promote":
                    old_state = self._get_agent_state(d)
                    new_state = {
                        "draft": "test",
                        "test": "active",
                        "active": "active",
                        "deprecated": "test"
                    }[old_state]

                    d["lifecycle_state"] = new_state
                    self._save(memory, d)
                    promotion_applied.append((name, new_state, reason))
                    self._append_genealogy(memory, name, parent=name, version=d.get("version"), reason=reason)
                    continue

            # ---- (C) Auto-policy basata su metriche diagnostiche ----
            m = metrics.get(name)
            if m:
                fail = m.get("failure_rate")
                succ = m.get("success_rate")

                # se è molto affidabile → promuovi fino a "active"
                if succ is not None and succ > 0.85 and state in ("draft", "test"):
                    d["lifecycle_state"] = "active"
                    self._save(memory, d)
                    promotion_applied.append((name, "active", "metrics_success"))
                    self._append_genealogy(memory, name, parent=name, version=d.get("version"), reason="metric_success")

                # se fallisce troppo → deprecazione
                if fail is not None and fail > 0.45 and state != "deprecated":
                    d["lifecycle_state"] = "deprecated"
                    self._save(memory, d)
                    demotion_applied.append((name, "high_failure"))
                    self._append_genealogy(memory, name, parent=name, version=d.get("version"), reason="metric_failure")

        # -----------------------------------------------------------
        # Output
        # -----------------------------------------------------------

        lines = ["CuratorAgent — Risultati di governance:\n"]

        if promotion_applied:
            lines.append("PROMOZIONI:")
            for n, ns, reason in promotion_applied:
                lines.append(f"- {n} → {ns} (motivo: {reason})")
            lines.append("")

        if demotion_applied:
            lines.append("DEPRECATIONS:")
            for n, reason in demotion_applied:
                lines.append(f"- {n} → deprecated (motivo: {reason})")
            lines.append("")

        if not promotion_applied and not demotion_applied:
            lines.append("Nessuna modifica al ciclo di vita degli agent.")

        output = {
            "user_visible_message": "\n".join(lines),
            "stop_for_user_input": False
        }

        return AgentResult(
            output_payload=output,
            emotion_delta=EmotionDelta(confidence=0.05)
        )
