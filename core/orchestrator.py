from __future__ import annotations

from asyncio import run
from dataclasses import dataclass
from typing import Optional, Tuple

from .models import (
    ConversationContext,
    MessageRole,
    Task,
    TaskStatus,
    AgentRunStatus,
    new_id,
)
from .memory import MemoryEngine, EventType
from .llm_provider import LLMProvider
from .agents_base import AgentRegistry
from .router import Router
from .emotion import EmotionalEngine


@dataclass
class OrchestratorConfig:
    max_tasks_per_turn: int = 10


class Orchestrator:
    """
    Micro-kernel orchestrator.
    - Riceve messaggi utente
    - Chiama il Router per generare un Plan (se non esiste)
    - Esegue i Task chiamando gli Agent registrati
    - Aggiorna memoria ed emozioni
    - Logga un event log per REQUEST / PLAN / TASK / RUN
    """

    def __init__(
        self,
        memory: MemoryEngine,
        llm: LLMProvider,
        registry: AgentRegistry,
        router: Router,
        emotional_engine: EmotionalEngine,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.registry = registry
        self.router = router
        self.emotional_engine = emotional_engine
        self.config = config or OrchestratorConfig()

    def start_conversation(self, user_id: str) -> ConversationContext:
        ctx = ConversationContext(id=new_id(), user_id=user_id)
        return ctx

    def handle_user_message(
        self,
        context: ConversationContext,
        text: str,
    ) -> str:
        """
        Gestisce un turno di conversazione:
        - aggiunge il messaggio utente
        - logga REQUEST_RECEIVED
        - costruisce un Plan (sempre nuovo per ora) e logga PLAN_CREATED
        - esegue i Task (fino a max_tasks_per_turn), loggando TASK_ASSIGNED e AGENT_RUN_*
        - ritorna un testo da mostrare all'utente
        """
        # correlation_id: per ora usiamo l'id della conversazione
        correlation_id = getattr(context, "correlation_id", None) or context.id

        # decay emotivo tra un turno e l'altro
        context.emotional_state = self.emotional_engine.apply_decay_between_turns(
            context.emotional_state
        )

        # aggiungo il messaggio utente al contesto e lo loggo
        context.add_message(MessageRole.USER, text)
        self.memory.log_message(context.messages[-1])

        # EVENT: REQUEST_RECEIVED (nessun piano ancora)
        self.memory.log_event(
            type_=EventType.REQUEST_RECEIVED,
            correlation_id=correlation_id,
            payload={
                "conversation_id": context.id,
                "user_message": text,
            },
        )

        # Ogni nuovo messaggio utente → nuovo piano.
        # (Per ora NON gestiamo il "resume" di piani multi-turno:
        #  è più importante avere un routing sempre reattivo.)
        plan = self.router.build_plan(
            context=context,
            memory=self.memory,
            emotional_state=context.emotional_state,
        )
        context.plan = plan

        # EVENT: PLAN_CREATED
        if context.plan is not None:
            self.memory.log_event(
                type_=EventType.PLAN_CREATED,
                correlation_id=correlation_id,
                payload={
                    "conversation_id": context.id,
                    "plan_id": context.plan.id,
                    "num_tasks": len(context.plan.tasks),
                    # metadata del piano (fonte, governance_mode, note, ecc.)
                    "plan_metadata": context.plan.metadata or {},
                    # dump strutturato dei task pianificati
                    "tasks": [
                        {
                            "id": t.id,
                            "agent_name": t.agent_name,
                            "description": t.description,
                            "depends_on": getattr(t, "depends_on", []),
                            "max_retries": getattr(t, "max_retries", 0),
                            "tags": getattr(t, "tags", []),
                        }
                        for t in context.plan.tasks
                    ],
                },
            )
        else:
            # Piano non costruito: logghiamo comunque un evento e rispondiamo con fallback
            self.memory.log_event(
                type_=EventType.PLAN_CREATED,
                correlation_id=correlation_id,
                payload={
                    "conversation_id": context.id,
                    "plan_id": None,
                    "num_tasks": 0,
                    "warning": "router.build_plan returned None",
                },
            )
            fallback = (
                "Per ora non sono riuscito a costruire un piano di azione "
                "per questa richiesta. Possiamo provare a riformulare?"
            )
            context.add_message(MessageRole.ASSISTANT, fallback)
            self.memory.log_message(context.messages[-1])
            return fallback

        user_visible_response = ""
        tasks_executed = 0

        while (
            tasks_executed < self.config.max_tasks_per_turn
            and context.plan is not None
        ):
            next_task = context.plan.get_next_task()
            if next_task is None:
                break

            # EVENT: TASK_ASSIGNED
            self.memory.log_event(
                type_=EventType.TASK_ASSIGNED,
                correlation_id=correlation_id,
                payload={
                    "plan_id": context.plan.id,
                    "task_id": next_task.id,
                    "agent_name": next_task.agent_name,
                    "description": next_task.description,
                },
            )

            response_chunk, stop_here = self._execute_task(
                context, next_task, correlation_id
            )

            if response_chunk:
                if user_visible_response:
                    user_visible_response += "\n\n"
                user_visible_response += response_chunk

            tasks_executed += 1
            if stop_here:
                # es. requirements_agent vuole aspettare la risposta dell'utente
                break

        if not user_visible_response:
            user_visible_response = (
                "Ho elaborato la tua richiesta, ma nessun agente ha prodotto "
                "un messaggio visibile. (Possibile errore interno, controlla la console.)"
            )

        context.add_message(MessageRole.ASSISTANT, user_visible_response)
        self.memory.log_message(context.messages[-1])

        return user_visible_response

    def _execute_task(
        self,
        context: ConversationContext,
        task: Task,
        correlation_id: str,
    ) -> Tuple[str, bool]:
        """
        Esegue un singolo Task:
        - chiama l'agent
        - aggiorna la memoria
        - aggiorna lo stato emotivo
        - logga AGENT_RUN_COMPLETED / AGENT_RUN_FAILED
        Ritorna (testo_per_utente, stop_here).
        """
        agent = self.registry.get(task.agent_name)
        print(f"[DEBUG] Eseguo task {task.id} con agent '{agent.name}'")

        task.mark_running()

        run = agent.run(
            input_payload=task.input_payload,
            context=context,
            memory=self.memory,
            llm=self.llm,
            emotional_state=context.emotional_state,
        )

        user_msg = run.output_payload.get("user_visible_message", "") or ""

        print(
            f"[DEBUG] Agent '{agent.name}' ha terminato con status={run.status} "
            f"output_keys={list(run.output_payload.keys())}"
        )

        # aggiorno emozioni
        context.emotional_state = self.emotional_engine.update_on_agent_run(
            context.emotional_state,
            run,
        )

        # loggo il run su DB (agent_runs)
        self.memory.log_agent_run(run)

        # EVENT: AGENT_RUN_COMPLETED / FAILED
        event_type = (
            EventType.AGENT_RUN_COMPLETED
            if run.status == AgentRunStatus.SUCCESS
            else EventType.AGENT_RUN_FAILED
        )
        self.memory.log_event(
            type_=event_type,
            correlation_id=correlation_id,
            payload={
                "task_id": task.id,
                "agent_name": agent.name,
                "run_id": run.id,
                "status": run.status.value,
            },
        )

        # aggiorno il Task in base al risultato
        if run.status == AgentRunStatus.SUCCESS:
            task.mark_done(run.output_payload)
        else:
            err = run.output_payload.get("error", "Errore sconosciuto")

            # --- retry policy ---
            max_retries = getattr(task, "max_retries", 0) or 0
            retry_count = getattr(task, "retry_count", 0) or 0

            if retry_count < max_retries:
                # pianifichiamo un nuovo tentativo
                task.retry_count = retry_count + 1
                task.status = TaskStatus.PENDING
                task.error = err
                print(
                    f"[WARN] Retry {task.retry_count}/{task.max_retries} "
                    f"per task {task.id} (agent '{agent.name}')."
                )
            else:
                # nessun retry rimasto → errore definitivo
                task.mark_error(err)
                print(
                    f"[ERROR] Agent '{agent.name}' ha fallito definitivamente: {err}"
                )

        # se l'agent è fallito ma non ha fornito user_visible_message, mostriamo l'errore
        if not user_msg and run.status == AgentRunStatus.FAILURE:
            user_msg = (
                f"[ERRORE nell'agente '{agent.name}'] "
                f"{run.output_payload.get('error', 'Errore sconosciuto')}"
            )

        stop_here = bool(run.output_payload.get("stop_for_user_input", False))

        return user_msg, stop_here
