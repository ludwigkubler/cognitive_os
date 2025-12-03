from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from .models import (
    Plan,
    Task,
    ConversationContext,
    EmotionalState,
    new_id,
)
from .memory import MemoryEngine
from .llm_provider import LLMProvider
from .agents_base import AgentRegistry


class Router:
    """
    Router/Planner.

    Se è disponibile un LLM reale:
      - costruisce un prompt con:
        * lista agent disponibili (nome + descrizione),
        * ultimo messaggio utente,
        * stato emotivo,
        * un piccolo estratto di memoria
      - si aspetta un JSON con un array `plan`:
        [
          {
            "agent": "requirements_agent",
            "description": "...",
            "input": { ... }
          },
          ...
        ]

    Se l'LLM non è disponibile o risponde male:
      - usa un routing euristico (versione estesa con DB / R / agent creation
        + agent sociali: memory, profiling, curiosità).
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        registry: Optional[AgentRegistry] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry

    def _build_meta_router_plan(
        self,
        context: ConversationContext,
        memory: MemoryEngine,
        emotional_state: EmotionalState,
    ) -> Plan:
        """
        Usa meta_router_agent come planner di alto livello:
        prende il meta_plan e lo converte in Plan/Task core.
        """
        if self.registry is None:
            raise RuntimeError("MetaRouter: registry non disponibile")

        try:
            meta_agent = self.registry.get("meta_router_agent")
        except KeyError as exc:
            raise RuntimeError("MetaRouter: agent 'meta_router_agent' non registrato") from exc

        # Nessun input obbligatorio: MetaRouter usa context + memory + emozioni
        run = meta_agent.run(
            input_payload={},
            context=context,
            memory=memory,
            llm=self.llm,
            emotional_state=emotional_state,
        )

        payload = run.output_payload or {}
        steps = payload.get("meta_plan") or []
        plan_id = payload.get("plan_id") or new_id()

        plan = Plan(id=plan_id)
        # metadata = governance, note, ecc.
        plan.metadata["governance_mode"] = bool(payload.get("governance_mode", False))
        plan.metadata["governance_reason"] = payload.get("governance_reason")
        plan.metadata["governance_targets"] = payload.get("governance_targets") or []
        plan.metadata["notes"] = payload.get("notes") or ""

        for step in steps:
            agent_name = step.get("agent")
            if not agent_name:
                continue

            task_id = step.get("id") or new_id()
            t = Task(
                id=task_id,
                description=step.get("description") or f"Step eseguito da {agent_name}",
                agent_name=agent_name,
                input_payload=step.get("input") or {},
                depends_on=step.get("depends_on") or [],
                max_retries=int(step.get("max_retries") or 0),
                cost_estimate=step.get("cost_estimate"),
                budget=step.get("budget"),
            )
            plan.add_task(t)

        if not plan.tasks:
            raise ValueError("MetaRouter: meta_plan vuoto o non valido")

        return plan
    # --------------------------------------------------
    # Entry point usato dall'Orchestrator
    # --------------------------------------------------
    def build_plan(
        self,
        context: ConversationContext,
        memory: MemoryEngine,
        emotional_state: EmotionalState,
    ) -> Plan:
        # Usiamo sempre l'ultimo messaggio utente per capire se ci sono comandi espliciti
        user_last = context.messages[-1].content if context.messages else ""
        user_last_lc = user_last.lower()

        # 0) Comandi espliciti verso agent sociali/profilo → usa SUBITO il piano euristico
        if any(
            kw in user_last_lc
            for kw in [
                "user_profile_agent",
                "profilo utente",
                "profilo interno",
                "aggiorna il mio profilo",
                "preference_learner_agent",
                "impara le mie preferenze",
                "impara le preferenze",
                "aggiorna le mie preferenze",
                "curiosity_question_agent",
                "fammi domande personali",
                "fammi 1 o 2 domande personali",
                "fammi qualche domanda su di me",
            ]
        ):
            return self._build_heuristic_plan(context)

        # 1) Se c'è MetaRouterAgent registrato, lo usiamo come planner principale
        if self.registry is not None:
            try:
                return self._build_meta_router_plan(
                    context=context,
                    memory=memory,
                    emotional_state=emotional_state,
                )
            except Exception as exc:
                print(f"[Router] MetaRouterAgent fallito, fallback: {exc}")

        # 2) Se non c'è LLM o registry → solo euristico
        if self.llm is None or self.registry is None:
            return self._build_heuristic_plan(context)

        # 3) Prova il vecchio piano LLM-based
        try:
            return self._build_llm_plan(context, memory, emotional_state)
        except Exception as exc:
            print(f"[Router] LLM-plan fallito, uso euristico: {exc}")
            return self._build_heuristic_plan(context)

    # --------------------------------------------------
    # 1) Piano LLM-based (con metriche + metadata)
    # --------------------------------------------------
    def _build_llm_plan(
        self,
        context: ConversationContext,
        memory: MemoryEngine,
        emotional_state: EmotionalState,
    ) -> Plan:
        # elenco agent disponibili con descrizione + metriche
        agents_meta: List[Dict[str, Any]] = []
        agent_metrics = memory.get_agent_metrics_from_diagnostics()

        if self.registry is not None:
            for name in self.registry.list_agents():
                try:
                    agent = self.registry.get(name)
                    meta: Dict[str, Any] = {
                        "name": agent.name,
                        "description": getattr(agent, "description", ""),
                    }

                    # allega metriche se disponibili
                    m = agent_metrics.get(agent.name)
                    if m:
                        meta["metrics"] = m
                        try:
                            failure_rate = float(m.get("failure_rate", 0.0))
                        except Exception:
                            failure_rate = 0.0
                        meta["reliability_score"] = max(0.0, 1.0 - failure_rate)

                    agents_meta.append(meta)
                except Exception:
                    continue

        # piccolo estratto di memoria conversazionale
        recent_messages = memory.get_recent_messages(limit=10)
        mem_snippet = "\n".join(
            f"[{m.role.value}] {m.content}" for m in recent_messages
        )

        user_last = context.messages[-1].content if context.messages else ""

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

        system_prompt = (
            "Sei un Router/Planner per un sistema multi-agent. "
            "Ricevi una lista di agent disponibili (nome + descrizione + eventuali metriche), "
            "l'ultima richiesta utente, uno stato emotivo corrente e un breve contesto. "
            "Devi restituire un piano JSON con i passi da eseguire.\n\n"
            "Per ogni agent puoi avere un campo 'metrics' con, ad esempio:\n"
            "- failure_rate (0.0–1.0): tasso di fallimento recente\n"
            "- total_runs: numero di esecuzioni recenti\n"
            "- avg_duration / global_avg_duration: tempi medi di esecuzione\n\n"
            "LINEE GUIDA:\n"
            "- Preferisci agent con failure_rate basso e total_runs sufficiente.\n"
            "- Evita, se possibile, agent con failure_rate molto alto (> 0.5).\n"
            "- Se un agent è molto più lento della media, usalo solo quando necessario.\n\n"
            "RISPOSTA OBBLIGATORIAMENTE in JSON valido, con questo schema MINIMO:\n"
            "{\n"
            '  \"plan\": [\n'
            "    {\n"
            '      \"agent\": \"nome_agent\",\n'
            '      \"description\": \"breve descrizione del sotto-task\",\n'
            '      \"input\": { ... },\n'
            '      \"depends_on\": [\"id_task_opzionali\"],   // opzionale\n'
            '      \"max_retries\": 0,                       // opzionale\n'
            '      \"cost_estimate\": 0.0                    // opzionale\n'
            "    }\n"
            "  ],\n"
            '  \"notes\": \"opzionale, spiegazione sintetica\",\n'
            '  \"governance_mode\": \"standard\" // opzionale, es. \"standard\" / \"safe_default\" / \"aggressive\"\n'
            "}\n"
            "Non aggiungere testo fuori dal JSON."
        )

        from .models import Message, MessageRole  # type: ignore

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(
                    {
                        "available_agents": agents_meta,
                        "user_request": user_last,
                        "emotional_state": emo_payload,
                        "memory_snippet": mem_snippet,
                    },
                    ensure_ascii=False,
                ),
            )
        ]

        raw = self.llm.generate(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=1024,
        )

        data = json.loads(raw)  # se fallisce → eccezione → fallback

        if "plan" not in data or not isinstance(data["plan"], list):
            raise ValueError("Router LLM: JSON senza campo 'plan' valido")

        # Plan con metadata: fonte, modalità, governance, note
        plan = Plan(
            id=new_id(),
            metadata={
                "source": "llm",
                "router_mode": "llm",
                "governance_mode": data.get("governance_mode", "standard"),
                "notes": data.get("notes"),
            },
        )

        for step in data["plan"]:
            agent_name = step.get("agent")
            if not agent_name:
                continue
            description = step.get("description", f"Step eseguito da {agent_name}")
            input_payload = step.get("input", {}) or {}

            t = Task(
                id=new_id(),
                description=description,
                agent_name=agent_name,
                input_payload=input_payload,
                depends_on=step.get("depends_on") or [],
                max_retries=int(step.get("max_retries") or 0),
                cost_estimate=step.get("cost_estimate"),
                budget=step.get("budget"),
            )
            plan.add_task(t)

        # se il piano è vuoto, solleva per far scattare il fallback
        if not plan.tasks:
            raise ValueError("Router LLM: piano vuoto")

        return plan
    
    # --------------------------------------------------
    # 2) Piano euristico (esteso con agent sociali/profilo)
    # --------------------------------------------------
    def _build_heuristic_plan(self, context: ConversationContext) -> Plan:
        # metadata esplicita: piano generato a regole
        plan = Plan(
            id=new_id(),
            metadata={
                "source": "heuristic",
                "router_mode": "heuristic",
                # default prudente: niente governance aggressiva
                "governance_mode": "safe_default",
            },
        )

        user_last = (
            context.messages[-1].content if context.messages else ""
        )
        user_last_lc = user_last.lower()

        description_prefix = "Piano generato (euristico)"

        # --------------------------------------------------
        # Caso 0: COMANDI ESPLICITI per profiling / curiosità
        # --------------------------------------------------
        # 0.a - Comando diretto per user_profile_agent
        if (
            "user_profile_agent" in user_last_lc
            or "profilo utente" in user_last_lc
            or "profilo interno" in user_last_lc
            or "aggiorna il mio profilo" in user_last_lc
        ):
            t = Task(
                id=new_id(),
                description=f"{description_prefix}: aggiornamento profilo utente",
                agent_name="user_profile_agent",
                input_payload={
                    "user_message": user_last,
                    # il vero user_id viene letto da context.user_id
                    "max_messages": 40,
                    "max_user_memories": 80,
                },
            )
            plan.add_task(t)
            return plan

        # 0.b - Comando diretto per preference_learner_agent
        if (
            "preference_learner_agent" in user_last_lc
            or "impara le mie preferenze" in user_last_lc
            or "impara le preferenze" in user_last_lc
            or "aggiorna le mie preferenze" in user_last_lc
        ):
            t = Task(
                id=new_id(),
                description=f"{description_prefix}: apprendimento preferenze esplicito",
                agent_name="preference_learner_agent",
                input_payload={
                    "user_message": user_last,
                },
            )
            plan.add_task(t)
            return plan

        # 0.c - Comando diretto per curiosity_question_agent
        if (
            "curiosity_question_agent" in user_last_lc
            or "fammi domande personali" in user_last_lc
            or "fammi 1 o 2 domande personali" in user_last_lc
            or "fammi qualche domanda su di me" in user_last_lc
        ):
            t = Task(
                id=new_id(),
                description=f"{description_prefix}: domande di curiosità personali",
                agent_name="curiosity_question_agent",
                input_payload={
                    "user_message": user_last,
                    # user_profile_agent leggerà il profilo da memoria
                },
            )
            plan.add_task(t)
            return plan

        # --------------------------------------------------
        # Caso 1: richieste su database / schema / tabella
        # --------------------------------------------------
        if any(kw in user_last_lc for kw in ["database", "schema", "tabella", "tabelle"]):
            t1 = Task(
                id=new_id(),
                description=f"{description_prefix}: progettazione database",
                agent_name="database_designer_agent",
                input_payload={"user_message": user_last},
            )
            plan.add_task(t1)

            t2 = Task(
                id=new_id(),
                description=f"{description_prefix}: spiegazione schema DB",
                agent_name="explanation_agent",
                input_payload={},
            )
            plan.add_task(t2)
            return plan

        # --------------------------------------------------
        # Caso 2: richieste di analisi / churn / modello
        # --------------------------------------------------
        if any(
            kw in user_last_lc
            for kw in ["churn", "modello", "predict", "prevedere", "classificazione"]
        ):
            t1 = Task(
                id=new_id(),
                description=f"{description_prefix}: analisi in R (demo churn)",
                agent_name="r_analysis_agent",
                input_payload={
                    "analysis_type": "churn_demo",
                    "params": {},
                },
            )
            plan.add_task(t1)

            t2 = Task(
                id=new_id(),
                description=f"{description_prefix}: spiegazione risultati analisi",
                agent_name="explanation_agent",
                input_payload={},
            )
            plan.add_task(t2)
            return plan

        # --------------------------------------------------
        # Caso 3: stato hardware / pc
        # --------------------------------------------------
        if any(
            kw in user_last_lc
            for kw in [
                "stato del pc",
                "stato del computer",
                "stato hardware",
                "hardware",
                "temperature",
                "temperatura",
                "cpu",
                "ram",
                "disco",
                "vram",
                "gpu",
            ]
        ):
            t = Task(
                id=new_id(),
                description=f"{description_prefix}: lettura stato hardware",
                agent_name="hardware_agent",
                input_payload={},  # non parametrico
            )
            plan.add_task(t)
            return plan

        # --------------------------------------------------
        # Caso 4: richieste di archiviazione / riassunto memoria
        # --------------------------------------------------
        if "memoria" in user_last_lc and any(
            kw in user_last_lc
            for kw in ["riassumi", "archivia", "compatta", "riassunto"]
        ):
            scope = "conversation"
            if "globale" in user_last_lc or "global" in user_last_lc:
                scope = "global"
            elif "progetto" in user_last_lc or "project" in user_last_lc:
                scope = "project"
            elif "utente" in user_last_lc or "user" in user_last_lc:
                scope = "user"

            t = Task(
                id=new_id(),
                description=f"{description_prefix}: archiviazione/sintesi memoria",
                agent_name="archivist_agent",
                input_payload={
                    "scope": scope,
                },
            )
            plan.add_task(t)
            return plan

        # --------------------------------------------------
        # Caso 5: l'utente chiede lo stato interno / emotivo
        # --------------------------------------------------
        if any(
            kw in user_last_lc
            for kw in [
                "come stai",
                "come ti senti",
                "come vi sentite",
                "stato emotivo",
                "stato interno",
                "come va dentro",
                "come va dentro di te",
            ]
        ):
            t = Task(
                id=new_id(),
                description=f"{description_prefix}: spiegazione stato interno",
                agent_name="state_explainer_agent",
                input_payload={},  # per ora nessun parametro
            )
            plan.add_task(t)
            return plan

        # --------------------------------------------------
        # Caso 6: memorizzazione esplicita ("ricordati che...")
        # --------------------------------------------------
        if any(
            kw in user_last_lc
            for kw in [
                "ricordati che",
                "ricorda che",
                "salva in memoria",
                "memorizza",
                "memorizzare",
                "annota",
                "segna",
                "prendi nota",
            ]
        ):
            # Proviamo a ripulire la frase per estrarre solo il contenuto
            note = user_last
            prefixes = [
                "ricordati che",
                "ricorda che",
                "salva in memoria",
                "memorizza che",
                "memorizza",
                "annota",
                "segna che",
                "segna",
                "prendi nota",
            ]
            for pref in prefixes:
                idx = note.lower().find(pref)
                if idx != -1:
                    note = note[idx + len(pref) :].strip(" :.-")
                    break

            if not note:
                note = user_last

            t = Task(
                id=new_id(),
                description=f"{description_prefix}: memorizzazione esplicita",
                agent_name="memory_agent",
                input_payload={
                    "content": note,
                    # default: memorizzazione legata all'utente
                    "scope": "user",
                },
            )
            plan.add_task(t)
            return plan

        # --------------------------------------------------
        # Caso 7: contesto/riassunto del progetto
        # --------------------------------------------------
        if any(
            kw in user_last_lc
            for kw in ["contesto progetto", "contesto del progetto", "riassumi il progetto"]
        ):
            t = Task(
                id=new_id(),
                description=f"{description_prefix}: aggiornamento contesto progetto",
                agent_name="project_context_agent",
                input_payload={
                    "project_name": "cognitive_os",
                    "mode": "update",
                    "extra_notes": user_last,
                },
            )
            plan.add_task(t)
            return plan

        # --------------------------------------------------
        # Caso 8: richiesta di EDA esplicita sul dataset
        # --------------------------------------------------
        if any(kw in user_last_lc for kw in ["eda", "analisi esplorativa", "esplorativa"]):
            t1 = Task(
                id=new_id(),
                description=f"{description_prefix}: EDA in R",
                agent_name="r_eda_agent",
                input_payload={
                    "dataset_ref": {
                        "type": "csv",
                        "path": "/percorso/assoluto/al/tuo_file.csv",
                    }
                },
            )
            plan.add_task(t1)

            t2 = Task(
                id=new_id(),
                description=f"{description_prefix}: spiegazione risultati EDA",
                agent_name="explanation_agent",
                input_payload={},
            )
            plan.add_task(t2)
            return plan

        # --------------------------------------------------
        # Caso 9: creazione nuovi agent
        # --------------------------------------------------
        if "nuovo agente" in user_last_lc or "nuovi agent" in user_last_lc:
            # pipeline: architect -> validator -> security_review -> critic
            t1 = Task(
                id=new_id(),
                description=f"{description_prefix}: design nuovo agent",
                agent_name="architect_agent",
                input_payload={"user_message": user_last},
            )
            plan.add_task(t1)

            t2 = Task(
                id=new_id(),
                description=f"{description_prefix}: validazione definizione",
                agent_name="validator_agent",
                input_payload={},
            )
            plan.add_task(t2)

            t3 = Task(
                id=new_id(),
                description=f"{description_prefix}: security review",
                agent_name="security_review_agent",
                input_payload={},
            )
            plan.add_task(t3)

            t4 = Task(
                id=new_id(),
                description=f"{description_prefix}: promozione/attivazione",
                agent_name="critic_agent",
                input_payload={},
            )
            plan.add_task(t4)
            return plan

        # --------------------------------------------------
        # Caso 10: testo chiaramente personale / preferenze / storia
        #        → profiling + curiosità automatici
        # --------------------------------------------------
        if any(
            kw in user_last_lc
            for kw in [
                "mi chiamo",
                "sono nato",
                "lavoro a",
                "lavoro al",
                "lavoro presso",
                "mia figlia",
                "mio figlio",
                "mi piace",
                "non mi piace",
                "odio",
                "adoro",
            ]
        ):
            # 1) impara preferenze dal messaggio
            t1 = Task(
                id=new_id(),
                description=f"{description_prefix}: apprendimento preferenze dal testo",
                agent_name="preference_learner_agent",
                input_payload={
                    "user_message": user_last,
                },
            )
            plan.add_task(t1)

            # 2) genera (eventualmente) una domanda di curiosità
            t2 = Task(
                id=new_id(),
                description=f"{description_prefix}: domanda/i di curiosità personali",
                agent_name="curiosity_question_agent",
                input_payload={
                    "user_message": user_last,
                },
            )
            plan.add_task(t2)
            return plan

        # --------------------------------------------------
        # Default: conversazione generale → chat_agent
        # --------------------------------------------------
        t = Task(
            id=new_id(),
            description=f"{description_prefix}: dialogo generico",
            agent_name="chat_agent",
            input_payload={"user_message": user_last},
        )
        plan.add_task(t)
        return plan

