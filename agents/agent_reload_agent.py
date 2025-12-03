# agents/agent_reload_agent.py
from __future__ import annotations

from typing import Any, Dict

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


class AgentReloadAgent(Agent):
    """
    Prova a ricaricare dinamicamente gli agent da `agents/` e `r_agents/`
    usando `core.agent_loader.load_agents_from_packages`.

    Modalità:

    - mode="runtime" (default):
        * se esiste core.agents_base.ACTIVE_REGISTRY → chiama load_agents_from_packages(...)
          e ricarica gli agent in questo processo.
        * altrimenti → non fa runtime reload, ma pianifica il reload al prossimo riavvio.

    - mode="next_restart":
        * non prova nessun runtime reload,
          scrive solo una memoria "pending_agent_reload" per il prossimo avvio.

    Parametri input_payload:
      - mode: "runtime" | "next_restart" (default: "runtime")
      - dry_run: bool (default False)
      - packages: lista di package da scandire (default ["agents", "r_agents"])
    """

    name = "agent_reload_agent"
    description = "Gestisce il reload (runtime o next-restart) degli agent dal filesystem."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        mode = input_payload.get("mode", "runtime")
        dry_run = bool(input_payload.get("dry_run", False))
        packages = input_payload.get("packages") or ["agents", "r_agents"]

        # messaggi da restituire all'utente
        messages: list[str] = []

        # ----------------------------------------------------------
        # 1) Tentativo di runtime reload (se richiesto)
        # ----------------------------------------------------------
        runtime_reloaded = False
        num_before = None
        num_after = None

        if mode == "runtime" and not dry_run:
            try:
                # Importiamo loader e cerchiamo un registry globale opzionale
                from core import agent_loader  # type: ignore
                import core.agents_base as agents_base  # type: ignore

                registry = getattr(agents_base, "ACTIVE_REGISTRY", None)

                if registry is None:
                    messages.append(
                        "⚠️ Nessun ACTIVE_REGISTRY trovato in core.agents_base; "
                        "non posso ricaricare gli agent a runtime."
                    )
                else:
                    # conteggio prima
                    if hasattr(registry, "list_agents"):
                        num_before = len(registry.list_agents())
                    else:
                        num_before = None

                    agent_loader.load_agents_from_packages(registry, packages)
                    runtime_reloaded = True

                    if hasattr(registry, "list_agents"):
                        num_after = len(registry.list_agents())
                    else:
                        num_after = None

                    if num_before is not None and num_after is not None:
                        messages.append(
                            f"✅ Runtime reload completato: agent registrati {num_before} → {num_after} "
                            f"(packages: {packages})."
                        )
                    else:
                        messages.append(
                            f"✅ Runtime reload completato sui package: {packages} "
                            "(numero agent prima/dopo non disponibile)."
                        )

            except Exception as exc:  # noqa: BLE001
                messages.append(f"❌ Errore durante il runtime reload: {exc}")

        elif mode == "runtime" and dry_run:
            messages.append(
                f"Modalità dry_run: simulerei un runtime reload per i package {packages}, "
                "ma non sto modificando il registry."
            )

        # ----------------------------------------------------------
        # 2) Se runtime non disponibile / non usato → pianifico next-restart
        # ----------------------------------------------------------
        if (mode == "next_restart") or (mode == "runtime" and not runtime_reloaded):
            # Scriviamo una memoria procedurale che segnala la necessità di reload
            note = {
                "mode": mode,
                "packages": packages,
                "runtime_reloaded": runtime_reloaded,
                "dry_run": dry_run,
            }
            try:
                memory.store_item(
                    scope=MemoryScope.GLOBAL,
                    type_=MemoryType.PROCEDURAL,
                    key="pending_agent_reload",
                    content="Agent reload richiesto; effettuare reload all'avvio.",
                    metadata=note,
                )
                messages.append(
                    "ℹ️ Ho registrato in memoria una richiesta di reload agent "
                    "(pending_agent_reload) per il prossimo avvio."
                )
            except Exception as exc:  # noqa: BLE001
                messages.append(
                    f"⚠️ Non sono riuscito a scrivere la richiesta di reload in memoria: {exc}"
                )

        # ----------------------------------------------------------
        # 3) Messaggio finale per l'utente
        # ----------------------------------------------------------
        header = "🔄 AgentReloadAgent"
        lines = [header, ""]
        lines.extend(messages)
        user_msg = "\n".join(lines)

        output = {
            "user_visible_message": user_msg,
            "stop_for_user_input": False,
            "runtime_reloaded": runtime_reloaded,
            "num_before": num_before,
            "num_after": num_after,
            "packages": packages,
            "mode": mode,
            "dry_run": dry_run,
        }

        delta = EmotionDelta(
            curiosity=0.01,
            confidence=0.02 if runtime_reloaded else 0.0,
        )
        return AgentResult(output_payload=output, emotion_delta=delta)
