# agents/meta_router_agent.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from core.agents_base import Agent, AgentResult, ACTIVE_REGISTRY
from core.models import (
    EmotionalState,
    EmotionDelta,
    EventType,
    ConversationContext,
    Message,
    MessageRole,
    MemoryScope,
    MemoryType,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider
from core.models import new_id  # Plan/Task li userà l'Orchestrator quando converte il meta_plan


def _safe_json_loads(raw: str) -> Optional[dict]:
    """
    Stessa logica usata in altri agent LLM-based:
    - prova json.loads diretto,
    - se fallisce, prova a estrarre il primo blocco {...}.
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


class MetaRouterAgent(Agent):
    """
    MetaRouterAgent 2.0 / Planner evoluto LLM + euristiche + governance.

    Scopi:
      - Leggere stato agent (metriche Diagnostics, fallimenti, durata).
      - Leggere contesto (ultima richiesta utente, memorie recenti, stato emotivo).
      - Costruire un PIANO operativo multi-agent per la richiesta utente, quando normale.
      - Riconoscere quando serve una pipeline di governance/evoluzione:
          Architect → SecurityReview → Validator → Critic → Curator → Codegen
        e costruire un piano dedicato (governance_mode=True).
      - Lasciare all'Orchestrator solo l'esecuzione step-by-step dei Task.

    input_payload (opzionale):

      {
        "max_steps": 8,
        "force_target_agent": "nome_agent_opzionale",
        "force_simple": false,           # se true, piano minimale senza dipendenze
        "force_governance": false,       # se true, attiva esplicitamente pipeline governance
        "governance_target_agent": "...",
      }

    output_payload:

      {
        "user_visible_message": "...",
        "meta_plan": [...],     # lista di step (agent + input + dipendenze)
        "notes": "eventuali note LLM",
        "plan_id": "<uuid-like>",
        "governance_mode": bool,
        "governance_reason": "string",
        "governance_targets": ["agent1", ...],
        "stop_for_user_input": false
      }
    """

    name = "meta_router_agent"
    description = (
        "Planner evoluto che costruisce un piano multi-agent (LLM + metriche interne + governance), "
        "lasciando all'Orchestrator solo l'esecuzione sequenziale."
    )

    # ------------------------------------------------------------------ #
    # Helpers per governance
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_governance_intent(
        user_last: str,
        input_payload: Dict[str, Any],
        metrics: Dict[str, Dict[str, Any]],
        emotional_state: EmotionalState,
    ) -> Tuple[bool, str, List[str]]:
        """
        Decide se entrare in governance_mode e su quali agent.
        Restituisce:
          (governance_mode, governance_reason, governance_targets)
        """
        text = (user_last or "").lower()

        # (1) Forzato da input_payload
        if bool(input_payload.get("force_governance", False)):
            target = input_payload.get("governance_target_agent")
            targets = [target] if isinstance(target, str) and target else []
            return True, "force_governance_flag", targets

        # (2) Intent esplicito dall'utente
        gov_keywords = [
            "nuovo agent",
            "nuovo agente",
            "crea un agent",
            "crea un agente",
            "migliora l'agent",
            "migliora l'agente",
            "refactor",
            "rifattorizza",
            "rifattorizzare",
            "migliora il codice",
            "ottimizza il sistema",
            "governance",
            "architettura",
            "auto-miglioramento",
            "self improvement",
            "self-improvement",
        ]
        if any(kw in text for kw in gov_keywords):
            return True, "explicit_user_governance_request", []

        # (3) Trigger da metriche (ripetuti fallimenti)
        bad_candidates: List[str] = []
        for agent_name, m in metrics.items():
            try:
                failure_rate = float(m.get("failure_rate", 0.0))
            except Exception:
                failure_rate = 0.0
            try:
                total_runs = int(m.get("total_runs", 0))
            except Exception:
                total_runs = 0

            # condizioni "forti": molti run + alta failure
            if total_runs >= 5 and failure_rate >= 0.6:
                bad_candidates.append(agent_name)

        # se non ci sono candidati, niente governance automatica
        if not bad_candidates:
            return False, "", []

        # (4) Stato emotivo come moltiplicatore di urgenza
        #     Più è alta la frustrazione, più è legittimo attivare governance
        if emotional_state.frustration < 0.4:
            # problemi ci sono, ma non ancora tale da scatenare auto-governance
            return False, "", []

        reason = "high_failure_rate_agents"
        return True, reason, bad_candidates

    @staticmethod
    def _get_security_targets(memory: MemoryEngine) -> List[str]:
        """
        Cerca alert di sicurezza per individuare agent da mettere in governance_target.
        """
        targets: List[str] = []
        try:
            items = memory.find_items_by_key("security_alert")
        except Exception:
            return targets

        for it in items:
            try:
                content = it.content
                if isinstance(content, str):
                    obj = json.loads(content)
                elif isinstance(content, dict):
                    obj = content
                else:
                    continue
                ag = obj.get("agent")
                if isinstance(ag, str) and ag not in targets:
                    targets.append(ag)
            except Exception:
                continue
        return targets

    @staticmethod
    def _build_governance_plan(
        targets: List[str],
        max_steps: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Costruisce un piano di governance standard per i target:
          Architect → SecurityReview → Validator → Critic → Curator → Codegen
        """
        plan: List[Dict[str, Any]] = []
        notes = "Piano di governance automatica basato su metriche, sicurezza e/o richiesta esplicita."

        def add_step(agent: str, description: str, input_payload: Dict[str, Any]) -> None:
            if len(plan) >= max_steps:
                return
            plan.append(
                {
                    "agent": agent,
                    "description": description,
                    "input": input_payload,
                    "depends_on": [],
                    "max_retries": 0,
                    "cost_estimate": None,
                    "budget": None,
                }
            )

        # se non ci sono target, usiamo una governance generale
        if not targets:
            targets = []

        # ArchitectAgent: definizione/aggiornamento agent
        add_step(
            "architect_agent",
            "Definisce o aggiorna la configurazione degli agent candidati alla governance.",
            {"targets": targets},
        )

        # SecurityReviewAgent: controlla rischi
        add_step(
            "security_review_agent",
            "Esegue un controllo di sicurezza sui nuovi/aggiornati agent.",
            {"scan_all": False, "target_id": None},
        )

        # ValidatorAgent: controlli tecnici
        add_step(
            "validator_agent",
            "Verifica la correttezza tecnica delle definizioni di agent.",
            {"targets": targets},
        )

        # CriticAgent: revisione qualitativa
        add_step(
            "critic_agent",
            "Valuta la qualità degli agent candidati e suggerisce promozioni/deprecazioni.",
            {"target_agent": None, "lookback_runs": 80, "max_examples": 15},
        )

        # CuratorAgent: applica governance
        add_step(
            "curator_agent",
            "Applica le decisioni di governance (promozioni, deprecazioni) in base a critic/sicurezza/diagnostica.",
            {},
        )

        # CodegenAgent: genera/aggiorna codice agent
        add_step(
            "codegen_agent",
            "Genera o aggiorna il codice sorgente per le AgentDefinition approvate.",
            {"overwrite_existing": False, "dry_run": False},
        )

        return plan, notes

    @staticmethod
    def _has_requirements_sheet(
        context: ConversationContext,
        memory: MemoryEngine,
    ) -> bool:
        """
        Ritorna True se esiste una requirements_sheet associata a questa
        conversazione o al progetto corrente.
        """
        # 1) Scope CONVERSATION
        conv_id = getattr(context, "id", None)
        if conv_id:
            key_conv = f"requirements_sheet:{conv_id}"
            try:
                raw = memory.load_item_content(
                    key=key_conv,
                    scope=MemoryScope.CONVERSATION,
                    type_=MemoryType.PROCEDURAL,
                )
                if raw:
                    return True
            except Exception:
                pass

        # 2) Scope PROJECT
        project_id = getattr(context, "project_id", None) or getattr(
            context, "current_project_id", None
        )
        if project_id:
            key_proj = f"requirements_sheet:{project_id}"
            try:
                raw = memory.load_item_content(
                    key=key_proj,
                    scope=MemoryScope.PROJECT,
                    type_=MemoryType.PROCEDURAL,
                )
                if raw:
                    return True
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------ #
    # _run_impl
    # ------------------------------------------------------------------ #

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        max_steps = int(input_payload.get("max_steps", 8))
        max_steps = max(1, min(max_steps, 20))
        force_target_agent = input_payload.get("force_target_agent")
        force_simple = bool(input_payload.get("force_simple", False))

        # ----------------------------
        # 1) Costruisco metadata sugli agent
        # ----------------------------
        agents_meta: List[Dict[str, Any]] = []
        metrics = memory.get_agent_metrics_from_diagnostics()  # failure_rate, avg_duration, total_runs, ecc.

        if ACTIVE_REGISTRY is not None:
            for name in ACTIVE_REGISTRY.list_agents():
                try:
                    ag = ACTIVE_REGISTRY.get(name)
                except Exception:
                    continue

                meta: Dict[str, Any] = {
                    "name": ag.name,
                    "description": getattr(ag, "description", ""),
                }

                m = metrics.get(ag.name)
                if m:
                    meta["metrics"] = m
                    try:
                        failure_rate = float(m.get("failure_rate", 0.0))
                    except Exception:
                        failure_rate = 0.0
                    meta["reliability_score"] = max(0.0, 1.0 - failure_rate)

                agents_meta.append(meta)

        # ----------------------------
        # 2) Contesto: ultimo messaggio utente + snippet memoria
        # ----------------------------
        user_last = context.messages[-1].content if context.messages else ""

        recent_messages = memory.get_recent_messages(limit=10)
        mem_snippet = "\n".join(
            f"[{m.role.value}] {m.content}" for m in recent_messages
        )

        emo_payload = {
            "curiosity": emotional_state.curiosity,
            "confidence": emotional_state.confidence,
            "fatigue": emotional_state.fatigue,
            "frustration": emotional_state.frustration,
            "mood": emotional_state.mood,
            "energy": emotional_state.energy,
            "playfulness": emotional_state.playfulness,
            "social_need": emotional_state.social_need,
            "learning_drive": emotional_state.learning_drive,
        }

        # ----------------------------
        # 3) Governance detection
        # ----------------------------
        governance_mode, governance_reason, gov_targets_from_metrics = self._detect_governance_intent(
            user_last=user_last,
            input_payload=input_payload,
            metrics=metrics,
            emotional_state=emotional_state,
        )

        gov_targets_from_security = self._get_security_targets(memory)
        governance_targets = list(
            {t for t in (gov_targets_from_metrics + gov_targets_from_security) if t}
        )

        # 3.1) Se siamo in governance_mode → costruiamo SUBITO un piano di governance
        if governance_mode:
            plan_list, gov_notes = self._build_governance_plan(
                targets=governance_targets,
                max_steps=max_steps,
            )
            # clamp numero di step
            if len(plan_list) > max_steps:
                plan_list = plan_list[:max_steps]

            internal_plan_id = new_id()
            enriched_plan: List[Dict[str, Any]] = []
            for idx, step in enumerate(plan_list, start=1):
                agent_name = step.get("agent")
                if not agent_name:
                    continue
                step_id = step.get("id") or f"{internal_plan_id}_step_{idx}"
                enriched_plan.append(
                    {
                        "id": step_id,
                        "agent": agent_name,
                        "description": step.get("description") or f"Step eseguito da {agent_name}",
                        "input": step.get("input") or {},
                        "depends_on": step.get("depends_on") or [],
                        "max_retries": int(step.get("max_retries") or 0),
                        "cost_estimate": step.get("cost_estimate"),
                        "budget": step.get("budget"),
                    }
                )

            # Log dell'evento di piano creato (governance)
            try:
                memory.log_event(
                    type_=EventType.PLAN_CREATED,
                    correlation_id=context.conversation_id,
                    payload={
                        "plan_id": internal_plan_id,
                        "meta_plan": enriched_plan,
                        "governance_mode": True,
                        "governance_reason": governance_reason,
                        "governance_targets": governance_targets,
                        "user_id": context.user_id,
                    },
                )
            except Exception:
                pass

            user_msg = (
                "Sto avviando una pipeline di governance sugli agent esistenti "
                "per migliorare sicurezza e qualità complessiva."
            )

            output = {
                "user_visible_message": user_msg,
                "meta_plan": enriched_plan,
                "notes": gov_notes,
                "plan_id": internal_plan_id,
                "governance_mode": True,
                "governance_reason": governance_reason,
                "governance_targets": governance_targets,
                "stop_for_user_input": False,
            }

            delta = EmotionDelta(
                curiosity=0.03,
                confidence=0.04,
            )
            return AgentResult(output_payload=output, emotion_delta=delta)

        # 3.2) Se NON siamo in governance e NON esiste ancora una requirements_sheet,
        #      la prima cosa da fare è raccogliere i requisiti in modo strutturato.
        if not self._has_requirements_sheet(context, memory):
            internal_plan_id = new_id()
            step_id = f"{internal_plan_id}_step_1"

            req_step = {
                "id": step_id,
                "agent": "requirements_agent",
                "description": (
                    "Chiarire e strutturare i requisiti a partire "
                    "dall'ultima richiesta utente."
                ),
                "input": {"user_message": user_last},
                "depends_on": [],
                "max_retries": 0,
                "cost_estimate": None,
                "budget": None,
            }

            user_msg = (
                "Prima di pianificare l'analisi voglio capire meglio i requisiti.\n"
                "Ti farò alcune domande mirate sul problema e sul dataset."
            )

            output = {
                "user_visible_message": user_msg,
                "meta_plan": [req_step],
                "notes": "Step iniziale di raccolta requisiti.",
                "plan_id": internal_plan_id,
                "governance_mode": False,
                "governance_reason": "",
                "governance_targets": [],
                "stop_for_user_input": False,
            }

            # Log dell'evento di piano creato (intake requisiti)
            try:
                memory.log_event(
                    type_=EventType.PLAN_CREATED,
                    correlation_id=context.conversation_id,
                    payload={
                        "plan_id": internal_plan_id,
                        "meta_plan": [req_step],
                        "governance_mode": False,
                        "governance_reason": "",
                        "governance_targets": [],
                        "user_id": context.user_id,
                    },
                )
            except Exception:
                pass

            delta = EmotionDelta(
                curiosity=0.05,
                confidence=0.02,
            )
            return AgentResult(output_payload=output, emotion_delta=delta)

        # -------------------------------------------------------------- #
        # 4) Planner LLM normale (richiesta utente “non governance”)
        # -------------------------------------------------------------- #
        planner_input = {
            "available_agents": agents_meta,
            "user_request": user_last,
            "emotional_state": emo_payload,
            "memory_snippet": mem_snippet,
            "max_steps": max_steps,
            "force_target_agent": force_target_agent,
            "force_simple": force_simple,
        }

        system_prompt = (
            "Sei un Meta-Router/Planner per un sistema multi-agent.\n"
            "Ricevi:\n"
            "- la lista di agent disponibili (nome, descrizione, metriche interne),\n"
            "- l'ultima richiesta utente,\n"
            "- uno stato emotivo corrente,\n"
            "- un breve contesto di conversazione recente.\n\n"
            "Devi costruire un PIANO operativo, come lista di step, dove ogni step "
            "specifica quale agent chiamare, con quali input, eventuali dipendenze, "
            "e quanti retry sono sensati.\n\n"
            "Usa questa logica:\n"
            "- Preferisci agent con failure_rate basso e sufficienti total_runs.\n"
            "- Evita agent con failure_rate molto alto (> 0.5) se ci sono alternative.\n"
            "- Se un agent è molto più lento della media (avg_duration >> global_avg_duration), "
            "usalo solo se davvero necessario.\n"
            "- Se force_target_agent è valorizzato, includi almeno uno step per quell'agent.\n"
            "- Rispetta max_steps come numero massimo di step nel piano.\n\n"
            "RISPOSTA OBBLIGATORIAMENTE in JSON valido, con questa struttura MINIMA:\n"
            "{\n"
            '  \"plan\": [\n'
            "    {\n"
            '      \"agent\": \"nome_agent\",\n'
            '      \"description\": \"breve descrizione del sotto-task\",\n'
            '      \"input\": { },\n'
            '      \"depends_on\": [\"id_task_opzionali\"],\n'
            '      \"max_retries\": 0,\n'
            '      \"cost_estimate\": 0.0,\n'
            '      \"budget\": null\n'
            "    }\n"
            "  ],\n"
            '  \"notes\": \"opzionale, spiegazione sintetica in italiano\"\n'
            "}\n"
            "Non aggiungere testo fuori dal JSON."
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(planner_input, ensure_ascii=False),
            )
        ]

        try:
            raw = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=1024,
            )
            data = _safe_json_loads(raw) or {}
        except Exception:
            data = {}

        plan_list = data.get("plan") or []
        notes = data.get("notes") or ""

        # se LLM ha fallito o dato plan vuoto → fallback euristico minimale
        if not isinstance(plan_list, list) or not plan_list:
            plan_list, notes = self._fallback_heuristic_plan(
                user_last=user_last,
                max_steps=max_steps,
            )

        # clamp numero di step
        if len(plan_list) > max_steps:
            plan_list = plan_list[:max_steps]

        # assegno id a ogni step così può essere convertito in Plan/Task vero
        internal_plan_id = new_id()
        enriched_plan: List[Dict[str, Any]] = []
        for idx, step in enumerate(plan_list, start=1):
            agent_name = step.get("agent")
            if not agent_name:
                continue
            step_id = step.get("id") or f"{internal_plan_id}_step_{idx}"

            enriched_plan.append(
                {
                    "id": step_id,
                    "agent": agent_name,
                    "description": step.get("description") or f"Step eseguito da {agent_name}",
                    "input": step.get("input") or {},
                    "depends_on": step.get("depends_on") or [],
                    "max_retries": int(step.get("max_retries") or 0),
                    "cost_estimate": step.get("cost_estimate"),
                    "budget": step.get("budget"),
                }
            )

        # Messaggio per l'utente
        if not enriched_plan:
            user_msg = (
                "MetaRouterAgent non è riuscito a costruire un piano operativo "
                "per la tua richiesta. Possiamo riprovare specificando meglio cosa vuoi ottenere."
            )
        else:
            bullets = []
            for s in enriched_plan:
                bullets.append(f"- {s['agent']}: {s['description']}")
            plan_text = "\n".join(bullets)

            if notes:
                user_msg = (
                    "Ho costruito un piano di lavoro multi-agent. In sintesi eseguirò:\n\n"
                    f"{plan_text}\n\n"
                    f"Note interne: {notes}"
                )
            else:
                user_msg = (
                    "Ho costruito un piano di lavoro multi-agent. In sintesi eseguirò:\n\n"
                    f"{plan_text}"
                )

        output = {
            "user_visible_message": user_msg,
            "meta_plan": enriched_plan,
            "notes": notes,
            "plan_id": internal_plan_id,
            "governance_mode": False,
            "governance_reason": "",
            "governance_targets": [],
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(
            confidence=0.04,
            curiosity=0.03,
        )
        return AgentResult(output_payload=output, emotion_delta=delta)

    # ------------------------------------------------------------------ #
    #  Fallback euristico (simile al Router euristico ma minimale)
    # ------------------------------------------------------------------ #

    def _fallback_heuristic_plan(
        self,
        user_last: str,
        max_steps: int,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Se l'LLM non dà un piano valido, usa alcune euristiche semplici
        basate sul contenuto dell'ultima richiesta.
        """
        text = (user_last or "").lower()
        plan: List[Dict[str, Any]] = []
        notes = "Piano generato in modalità euristica (LLM non disponibile o risposta non valida)."

        def add_step(agent: str, description: str, input_payload: Dict[str, Any]) -> None:
            if len(plan) >= max_steps:
                return
            plan.append(
                {
                    "agent": agent,
                    "description": description,
                    "input": input_payload,
                    "depends_on": [],
                    "max_retries": 0,
                    "cost_estimate": None,
                    "budget": None,
                }
            )

        # Caso DB
        if any(kw in text for kw in ["database", "schema", "tabella", "tabelle"]):
            add_step(
                "database_designer_agent",
                "Progettazione dello schema di database a partire dai requisiti descritti.",
                {"user_message": user_last},
            )
            add_step(
                "explanation_agent",
                "Spiegazione in linguaggio naturale dello schema proposto.",
                {},
            )
            return plan, notes

        # Caso churn / modello / classificazione
        if any(
            kw in text
            for kw in ["churn", "modello", "predict", "prevedere", "classificazione"]
        ):
            add_step(
                "r_analysis_agent",
                "Esecuzione di una demo di churn/analisi in R.",
                {"analysis_type": "churn_demo", "params": {}},
            )
            add_step(
                "explanation_agent",
                "Spiegazione dei risultati dell'analisi in linguaggio naturale.",
                {},
            )
            return plan, notes

        # Caso EDA esplicita
        if any(kw in text for kw in ["eda", "analisi esplorativa", "esplorativa"]):
            add_step(
                "r_eda_agent",
                "Analisi esplorativa dei dati in R (EDA generica).",
                {
                    "dataset_ref": {
                        "type": "csv",
                        "path": "/percorso/assoluto/al/tuo_file.csv",
                    }
                },
            )
            add_step(
                "explanation_agent",
                "Riassunto e spiegazione dei risultati EDA.",
                {},
            )
            return plan, notes

        # Profilo / preferenze / curiosità
        if any(
            kw in text
            for kw in ["mi chiamo", "sono nato", "mi piace", "non mi piace", "mia figlia", "mio figlio"]
        ):
            add_step(
                "preference_learner_agent",
                "Apprendimento delle preferenze personali dal testo dell'utente.",
                {"user_message": user_last},
            )
            add_step(
                "curiosity_question_agent",
                "Generazione di una o due domande personali per conoscerti meglio.",
                {"user_message": user_last},
            )
            return plan, notes

        # Default: requirements + planner + chat/explanation
        add_step(
            "requirements_agent",
            "Chiarire meglio i requisiti e l'obiettivo dell'utente.",
            {"user_message": user_last},
        )
        add_step(
            "analysis_planner_agent",
            "Proporre un piano analitico di alto livello.",
            {},
        )
        add_step(
            "explanation_agent",
            "Restituire una spiegazione coerente del piano e dei prossimi passi.",
            {},
        )
        return plan, notes


# Registrazione auto se ACTIVE_REGISTRY è disponibile
if ACTIVE_REGISTRY is not None:
    ACTIVE_REGISTRY.register(MetaRouterAgent())
