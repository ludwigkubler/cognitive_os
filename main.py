from __future__ import annotations

from core.memory import MemoryEngine
from core.llm_provider import GroqLLM
from core.agents_base import AgentRegistry
import core.agents_base as agents_base

from core.router import Router
from core.emotion import EmotionalEngine
from core.orchestrator import Orchestrator, OrchestratorConfig
from core.agent_loader import load_agents_from_packages


def build_orchestrator() -> Orchestrator:
    memory = MemoryEngine()
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    registry = AgentRegistry()
    agents_base.ACTIVE_REGISTRY = registry
    load_agents_from_packages(registry, ["agents", "r_agents"])
    router = Router(llm=llm, registry=registry)
    emotional_engine = EmotionalEngine()
    config = OrchestratorConfig(max_tasks_per_turn=10)

    orchestrator = Orchestrator(
        memory=memory,
        llm=llm,
        registry=registry,
        router=router,
        emotional_engine=emotional_engine,
        config=config,
    )
    return orchestrator


def run_cli() -> None:
    orchestrator = build_orchestrator()
    ctx = orchestrator.start_conversation(user_id="user-1")

    print("Sistema pronto")
    print("Scrivi quello che vuoi. Digita 'exit' o 'esci' per uscire.\n")

    while True:
        user_input = input("TU> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "esci", "fine", "stop", "q", "x", "end", "terminate"}:
            print("Faccio un pisolino...")
            break

        reply = orchestrator.handle_user_message(ctx, user_input)
        print("\nAI>")
        print(reply)
        print("\n---\n")

if __name__ == "__main__":
    run_cli()
