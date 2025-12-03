from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


DANGEROUS_KEYWORDS = [
    "rm -rf",
    "drop table",
    "format c:",
    "shutdown",
    "kill -9",
    "exec(",
    "eval(",
    "subprocess.Popen",
    "os.system(",
]


class SecurityReviewAgent(Agent):
    """
    SecurityReviewAgent 2.0

    - Controlla una o più AgentDefinition alla ricerca di pattern pericolosi.
    - In caso di rischio:
      - forza lifecycle_state = 'draft'
      - imposta is_active = False
      - scrive un alert strutturato in memoria (key='security_alert')
    - Scrive comunque l'esito in 'security_review_last' per compatibilità.
    """

    name = "security_review_agent"
    description = "Esegue un controllo di sicurezza sulle AgentDefinition e blocca quelle potenzialmente pericolose."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        if not hasattr(memory, "list_agent_definitions"):
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "SecurityReview: nessun registry persistente per agent_definitions; "
                        "controlli avanzati non disponibili."
                    ),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        defs = memory.list_agent_definitions()
        if not defs:
            return AgentResult(
                output_payload={
                    "user_visible_message": "SecurityReview: nessuna AgentDefinition da controllare.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        target_id: Optional[str] = input_payload.get("target_id")
        scan_all: bool = bool(input_payload.get("scan_all", False))

        # Se è specificato target_id, controlla solo quella; altrimenti:
        # - se scan_all=True → tutte
        # - else → solo l'ultima (comportamento originale)
        candidates: List[Dict[str, Any]] = []

        if target_id:
            for d in defs:
                if isinstance(d, dict) and d.get("id") == target_id:
                    candidates.append(d)
                    break
        elif scan_all:
            candidates = [d for d in defs if isinstance(d, dict)]
        else:
            last = defs[-1]
            if isinstance(last, dict):
                candidates.append(last)

        if not candidates:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "SecurityReview: nessuna AgentDefinition compatibile trovata "
                        f"(target_id={target_id!r})."
                    ),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        alerts: List[Dict[str, Any]] = []
        checked: List[str] = []

        for candidate in candidates:
            name = candidate.get("name", "<unknown>")
            desc = candidate.get("description", "")
            cfg = candidate.get("config", {}) or {}

            text_blobs: List[str] = [name, desc]
            for v in cfg.values():
                if isinstance(v, str):
                    text_blobs.append(v)

            full_text = "\n".join(text_blobs).lower()
            hits = [kw for kw in DANGEROUS_KEYWORDS if kw in full_text]

            checked.append(name)

            if hits:
                # blocca l'agent
                candidate["lifecycle_state"] = "draft"
                candidate["is_active"] = False
                candidate.setdefault("security_flags", {})
                candidate["security_flags"]["dangerous_keywords"] = hits
                try:
                    memory.save_agent_definition(candidate)
                except Exception:
                    pass

                # alert strutturato (consumato poi dal CuratorAgent)
                alert = {
                    "agent": name,
                    "agent_id": candidate.get("id"),
                    "hits": hits,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                alerts.append(alert)

                try:
                    memory.store_item(
                        scope=MemoryScope.GLOBAL,
                        type_=MemoryType.PROCEDURAL,
                        key="security_alert",
                        content=json.dumps(alert),
                    )
                except Exception:
                    pass

                verdict = (
                    f"❌ SecurityReview: l'agent '{name}' contiene pattern potenzialmente pericolosi: "
                    + ", ".join(hits)
                )
            else:
                verdict = f"✅ SecurityReview: nessun pattern pericoloso trovato per l'agent '{name}'."

            # ultimo esito (compatibilità con versioni precedenti)
            try:
                memory.store_item(
                    scope=MemoryScope.CONVERSATION,
                    type_=MemoryType.PROCEDURAL,
                    key="security_review_last",
                    content=json.dumps(
                        {
                            "agent_name": name,
                            "security_ok": not bool(hits),
                            "hits": hits,
                        }
                    ),
                )
            except Exception:
                pass

        # Messaggio per l'utente
        lines: List[str] = []
        if not alerts:
            lines.append(
                f"SecurityReview: ho controllato {len(checked)} agent, nessun rischio evidente trovato."
            )
        else:
            lines.append(
                f"SecurityReview: ho controllato {len(checked)} agent; "
                f"{len(alerts)} contengono pattern potenzialmente pericolosi."
            )
            lines.append("")
            lines.append("Agenti bloccati:")
            for al in alerts:
                lines.append(
                    f"- {al['agent']} (id={al.get('agent_id')}) → pattern: {', '.join(al['hits'])}"
                )

        output = {
            "user_visible_message": "\n".join(lines),
            "stop_for_user_input": False,
            "checked_agents": checked,
            "alerts": alerts,
        }

        delta = EmotionDelta(
            confidence=0.02 if not alerts else -0.02,
            frustration=0.02 if alerts else 0.0,
        )
        return AgentResult(output_payload=output, emotion_delta=delta)
