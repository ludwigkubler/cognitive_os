# agents/critic_agent.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult, ACTIVE_REGISTRY
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
    Message,
    MessageRole,
    EventType,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider

# Riutilizziamo il profilo utente per tarare il livello di dettaglio
try:
    from agents.user_profile_agent import _ensure_base_profile
except ImportError:  # fallback minimale
    def _ensure_base_profile(user_id: str, raw_profile: Optional[str]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": 1,
            "user_id": user_id,
            "display_name": user_id,
            "last_seen": now,
            "interaction_style": {
                "prefers_short_answers": False,
                "likes_technical_detail": True,
                "likes_humor": True,
                "sensitivity_level": "medium",
                "formality": "casual",
            },
        }


def _safe_json_loads(raw: str) -> Optional[dict]:
    """
    Come negli altri agent: prova json.loads, poi estrai il blocco {...}.
    """
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        snippet = raw[start:end]
        val2 = json.loads(snippet)
        if isinstance(val2, dict):
            return val2
    except Exception:
        return None

    return None


class CriticAgent(Agent):
    """
    CriticAgent / Reviewer + Governance advisor (2.0)

    Scopo:
      - Leggere memorie procedurali + output recenti di altri agent,
      - Valutare qualità dei risultati (success/failure, chiarezza, stabilità),
      - Suggerire eventuali re-run con parametri diversi,
      - Produrre suggerimenti di governance (promuovere/demotere agent) per CuratorAgent,
      - Scrivere feedback sia in diagnostic_alert globale,
        sia in memorie PROJECT (se disponibile).

    Input atteso (input_payload):
      {
        "target_agent": "nome_agent_opzionale",
        "lookback_runs": 40,      # quanti run considerare (clamp 10–200)
        "max_examples": 10,       # quante esecuzioni sintetizzare per l'LLM
        "include_plans": true     # se includere snapshot dei PLAN_CREATED
      }

    Output:
      - user_visible_message: testo leggibile per l’utente
      - summary: riassunto sintetico della review
      - quality_assessment: lista per agent con valutazione e problemi
      - rerun_suggestions: lista strutturata di suggerimenti di re-run
      - governance_suggestions: suggerimenti per CuratorAgent (promote/demote/keep)
      - diagnostic_memory_id: id della memoria GLOBAL/diagnostic_alert (se salvata)
      - project_feedback_memory_id: id memoria PROJECT (se salvata)
    """

    name = "critic_agent"
    description = (
        "Valuta la qualità dei risultati degli altri agent, suggerisce re-run "
        "con parametri diversi e produce suggerimenti di governance per CuratorAgent."
    )

    # ------------------------------------------------------------------ #
    #  Helpers interni
    # ------------------------------------------------------------------ #

    def _load_user_profile(self, context: ConversationContext, memory: MemoryEngine) -> Dict[str, Any]:
        user_id = getattr(context, "user_id", None) or "unknown"
        profile_key = f"user_profile:{user_id}"
        raw_profile = memory.load_item_content(
            key=profile_key,
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
        )
        return _ensure_base_profile(user_id=user_id, raw_profile=raw_profile)

    def _collect_runs_summary(
        self,
        memory: MemoryEngine,
        target_agent: Optional[str],
        lookback_runs: int,
        max_examples: int,
    ) -> List[Dict[str, Any]]:
        try:
            runs = memory.get_recent_agent_runs(limit=lookback_runs)
        except Exception:
            return []

        filtered: List[Dict[str, Any]] = []
        for r in reversed(runs):  # dal più recente al meno recente
            if target_agent and r.agent_name != target_agent:
                continue

            output = r.output_payload or {}
            content_snippet = ""

            if isinstance(output, dict):
                msg = output.get("user_visible_message")
                if isinstance(msg, str) and msg:
                    content_snippet = msg[:300]
                else:
                    raw = json.dumps(output, ensure_ascii=False)
                    content_snippet = raw[:300]

            filtered.append(
                {
                    "agent_name": r.agent_name,
                    "status": r.status.value,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "input_payload": r.input_payload,
                    "output_snippet": content_snippet,
                }
            )
            if len(filtered) >= max_examples:
                break

        return filtered

    def _collect_plan_snapshot(
        self,
        memory: MemoryEngine,
        context: ConversationContext,
        target_agent: Optional[str],
        limit_events: int = 200,
    ) -> Dict[str, Any]:
        """
        Estrae da events i PLAN_CREATED (e in futuro i TASK_ASSIGNED) per dare
        al Critic una vista sul routing/planning reale.
        """
        correlation_id = getattr(context, "correlation_id", None)
        try:
            events = memory.get_events(
                correlation_id=correlation_id,
                limit=limit_events,
            )
        except Exception:
            return {"plan_events": [], "agents_usage": {}}

        plan_events: List[Dict[str, Any]] = []
        agents_usage: Dict[str, Dict[str, Any]] = {}

        for ev in events:
            if ev.type == EventType.PLAN_CREATED:
                payload = ev.payload or {}
                tasks = payload.get("tasks", []) or []
                agents_in_plan: List[str] = []
                for t in tasks:
                    agent_name = t.get("agent") or t.get("agent_name")
                    if not agent_name:
                        continue
                    if target_agent and agent_name != target_agent:
                        # se filtriamo su un agent specifico, saltiamo gli altri
                        continue
                    agents_in_plan.append(agent_name)
                    stats = agents_usage.setdefault(
                        agent_name,
                        {"planned_count": 0, "last_seen_at": None},
                    )
                    stats["planned_count"] += 1
                    stats["last_seen_at"] = ev.timestamp.isoformat()

                plan_events.append(
                    {
                        "event_id": ev.id,
                        "timestamp": ev.timestamp.isoformat(),
                        "correlation_id": ev.correlation_id,
                        "governance_mode": payload.get("governance_mode"),
                        "source": payload.get("source", "unknown"),
                        "num_tasks": len(tasks),
                        "agents": agents_in_plan,
                    }
                )

        return {
            "plan_events": plan_events,
            "agents_usage": agents_usage,
        }

    def _find_last_security_review(self, memory: MemoryEngine) -> Optional[Dict[str, Any]]:
        """
        Cerca l'ultimo output di security_review_agent nei recenti agent_runs.
        """
        try:
            runs = memory.get_recent_agent_runs(limit=200)
        except Exception:
            return None

        for r in reversed(runs):
            if r.agent_name == "security_review_agent":
                if isinstance(r.output_payload, dict):
                    return r.output_payload
                break
        return None

    def _collect_agent_definitions_brief(self, memory: MemoryEngine) -> List[Dict[str, Any]]:
        """
        Snapshot compatto delle agent_definitions: serve all'LLM per proporre
        promozioni/demozioni coerenti con lifecycle_state + is_active.
        """
        if not hasattr(memory, "list_agent_definitions"):
            return []

        try:
            defs = memory.list_agent_definitions()
        except Exception:
            return []

        brief_list: List[Dict[str, Any]] = []
        for d in defs:
            created_at = d.get("created_at")
            if isinstance(created_at, datetime):
                created_at_str = created_at.isoformat()
            else:
                created_at_str = str(created_at) if created_at is not None else None

            brief_list.append(
                {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "description": d.get("description"),
                    "is_active": bool(d.get("is_active", False)),
                    "lifecycle_state": d.get("lifecycle_state", "draft"),
                    "parent_id": d.get("parent_id"),
                    "created_at": created_at_str,
                }
            )
        return brief_list

    # ------------------------------------------------------------------ #
    #  Core run
    # ------------------------------------------------------------------ #

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        # Parametri
        target_agent = input_payload.get("target_agent")
        lookback_runs = int(input_payload.get("lookback_runs", 40))
        max_examples = int(input_payload.get("max_examples", 10))
        include_plans = bool(input_payload.get("include_plans", True))

        lookback_runs = max(10, min(lookback_runs, 200))
        max_examples = max(3, min(max_examples, 30))

        # Profilo utente → livello di dettaglio
        user_profile = self._load_user_profile(context, memory)
        interaction = user_profile.get("interaction_style", {})
        prefers_short = bool(interaction.get("prefers_short_answers", False))
        likes_tech = bool(interaction.get("likes_technical_detail", True))

        # Metriche dai diagnostics (failure_rate, avg_duration, ecc.)
        agent_metrics = memory.get_agent_metrics_from_diagnostics()

        # Riassunto delle ultime esecuzioni
        runs_summary = self._collect_runs_summary(
            memory=memory,
            target_agent=target_agent,
            lookback_runs=lookback_runs,
            max_examples=max_examples,
        )

        if not runs_summary:
            msg = "CriticAgent: non ho esecuzioni recenti su cui fare una review."
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "summary": msg,
                    "quality_assessment": [],
                    "rerun_suggestions": [],
                    "governance_suggestions": [],
                    "diagnostic_memory_id": None,
                    "project_feedback_memory_id": None,
                },
                emotion_delta=EmotionDelta(confidence=-0.01),
            )

        # Snapshot dei piani (eventi PLAN_CREATED) e uso agent nei piani
        plan_snapshot: Dict[str, Any] = {"plan_events": [], "agents_usage": {}}
        if include_plans:
            plan_snapshot = self._collect_plan_snapshot(
                memory=memory,
                context=context,
                target_agent=target_agent,
            )

        # Ultimo security_review (se esiste)
        security_review_last = self._find_last_security_review(memory)

        # Definizioni agent (per lifecycle_state / is_active)
        agent_definitions_brief = self._collect_agent_definitions_brief(memory)

        # Costruiamo input per l'LLM
        llm_input = {
            "target_agent": target_agent,
            "user_profile": user_profile,
            "interaction_style": {
                "prefers_short_answers": prefers_short,
                "likes_technical_detail": likes_tech,
            },
            "agent_metrics": agent_metrics,
            "recent_runs": runs_summary,
            "plan_snapshot": plan_snapshot,
            "agent_definitions": agent_definitions_brief,
            "security_review_last": security_review_last,
        }

        detail_instruction = (
            "Scrivi una valutazione sintetica a bullet point, con poche frasi, "
            "evitando troppo dettaglio tecnico."
            if prefers_short and not likes_tech
            else "Puoi fornire un'analisi anche tecnica, con qualche dettaglio su errori, parametri e possibili miglioramenti."
        )

        system_prompt = (
            "Sei un revisore tecnico e di governance per un sistema multi-agent.\n"
            "Input:\n"
            "- metriche di diagnostica (failure_rate, total_runs, avg_duration) per ciascun agent,\n"
            "- esecuzioni recenti (recent_runs),\n"
            "- snapshot dei piani pianificati (plan_snapshot: PLAN_CREATED + agents_usage),\n"
            "- elenco sintetico delle agent_definitions (nome, lifecycle_state, is_active),\n"
            "- eventuale ultimo risultato di security_review_agent.\n\n"
            "Compiti:\n"
            "1) Valutare la qualità dei risultati (chiarezza, correttezza, stabilità) per ogni agent rilevante.\n"
            "2) Suggerire eventuali re-run con parametri diversi o agent alternativi.\n"
            "3) Produrre suggerimenti di governance per il CuratorAgent, in modo CONSERVATIVO:\n"
            "   - proponi 'promote' solo se failure_rate è basso e l'agent è usato nei piani,\n"
            "   - proponi 'demote' o 'deprecated' solo se ci sono forti segnali: molti fallimenti, warning di sicurezza.\n\n"
            "Rispondi OBBLIGATORIAMENTE con un JSON valido con struttura minima:\n"
            "{\n"
            '  \"summary\": \"breve riassunto generale in italiano\",\n'
            '  \"quality_assessment\": [\n'
            "    {\n"
            '      \"agent_name\": \"string\",\n'
            '      \"quality\": \"buona|media|problematica\",\n'
            '      \"issues\": [\"stringa\", \"...\"],\n'
            '      \"recommendations\": [\"stringa\", \"...\"]\n'
            "    }\n"
            "  ],\n"
            '  \"rerun_suggestions\": [\n'
            "    {\n"
            '      \"agent_name\": \"string\",\n'
            '      \"reason\": \"perché vale la pena rifare il run\",\n'
            '      \"suggested_params\": {\"chiave\": \"valore\"}\n'
            "    }\n"
            "  ],\n"
            '  \"governance_suggestions\": [\n'
            "    {\n"
            '      \"agent_name\": \"string\",\n'
            '      \"action\": \"promote|demote|keep\",\n'
            '      \"target_state\": \"draft|test|active|deprecated|null\",\n'
            '      \"confidence\": 0.0,\n'
            '      \"reason\": \"spiega brevemente il perché\"\n'
            "    }\n"
            "  ],\n"
            '  \"user_visible_message\": \"testo leggibile per l\\\'utente finale\"\n'
            "}\n\n"
            "REGOLE DI GOVERNANCE:\n"
            "- Sii molto cauto nel proporre 'deprecated': usalo solo per agent con fallimenti gravi/ricorrenti o segnalati da security_review.\n"
            "- Se non hai abbastanza informazioni per un agent, usa action=\"keep\" e target_state=null.\n"
            "- Non cambiare lo stato di agent che non compaiono né nei piani né nei run recenti.\n\n"
            f"{detail_instruction}\n"
            "Se non hai abbastanza informazioni per alcune sezioni, lasciale vuote."
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(llm_input, ensure_ascii=False),
            )
        ]

        try:
            llm_raw = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=900,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"CriticAgent: errore durante la chiamata all'LLM: {exc}"
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "summary": msg,
                    "quality_assessment": [],
                    "rerun_suggestions": [],
                    "governance_suggestions": [],
                    "diagnostic_memory_id": None,
                    "project_feedback_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.05, confidence=-0.05),
            )

        parsed = _safe_json_loads(llm_raw) or {}
        summary = str(parsed.get("summary") or "Review tecnica degli ultimi run completata.")
        user_msg = parsed.get("user_visible_message") or summary

        quality_assessment = parsed.get("quality_assessment") or []
        rerun_suggestions = parsed.get("rerun_suggestions") or []
        governance_suggestions = parsed.get("governance_suggestions") or []

        # ------------------------------------------------------------------
        # Scrittura feedback in memorie:
        #   - GLOBAL/PROCEDURAL/diagnostic_alert (come diagnostics_agent)
        #   - PROJECT/PROCEDURAL/project_review_feedback (se abbiamo project)
        # ------------------------------------------------------------------
        diagnostic_memory_id: Optional[int] = None
        project_feedback_memory_id: Optional[int] = None

        try:
            item = memory.store_item(
                scope=MemoryScope.GLOBAL,
                type_=MemoryType.PROCEDURAL,
                key="diagnostic_alert",
                content=user_msg,
                metadata={
                    "severity": "info",
                    "kind": "critic_review",
                    "target_agent": target_agent,
                    "has_rerun_suggestions": bool(rerun_suggestions),
                    "has_governance_suggestions": bool(governance_suggestions),
                },
            )
            diagnostic_memory_id = item.id
        except Exception:
            diagnostic_memory_id = None

        # scope progetto (se esiste un project_id nel contesto)
        project_id = getattr(context, "project_id", None) or getattr(context, "current_project_id", None)
        if project_id:
            try:
                item2 = memory.store_item(
                    scope=MemoryScope.PROJECT,
                    type_=MemoryType.PROCEDURAL,
                    key="project_review_feedback",
                    content=user_msg,
                    metadata={
                        "project_id": project_id,
                        "agent": self.name,
                        "target_agent": target_agent,
                    },
                )
                project_feedback_memory_id = item2.id
            except Exception:
                project_feedback_memory_id = None

        delta = EmotionDelta(
            curiosity=0.03,      # review → stimola miglioramento
            confidence=0.02,     # avere un quadro chiaro aumenta leggermente la fiducia
            frustration=0.0,
        )

        return AgentResult(
            output_payload={
                "user_visible_message": user_msg,
                "summary": summary,
                "quality_assessment": quality_assessment,
                "rerun_suggestions": rerun_suggestions,
                "governance_suggestions": governance_suggestions,
                "diagnostic_memory_id": diagnostic_memory_id,
                "project_feedback_memory_id": project_feedback_memory_id,
            },
            emotion_delta=delta,
        )


# Registrazione nell'ACTIVE_REGISTRY (usato da agent_loader/load_agents_from_packages)
if ACTIVE_REGISTRY is not None:
    ACTIVE_REGISTRY.register(CriticAgent())