from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import (
    AgentRun,
    AgentRunStatus,
    EmotionDelta,
    EmotionalState,
    ConversationContext,
    new_id,
)
from .memory import MemoryEngine
from .llm_provider import LLMProvider


@dataclass
class AgentResult:
    output_payload: Dict[str, Any]
    emotion_delta: EmotionDelta = field(default_factory=EmotionDelta)
    status: AgentRunStatus = AgentRunStatus.SUCCESS


class Agent(ABC):
    """
    Base class per tutti gli agent.
    Monofunzionale: uno scopo chiaro, input_payload -> output_payload.
    """

    name: str = "base_agent"
    description: str = "Base agent"

    def run(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentRun:
        try:
            result = self._run_impl(
                input_payload=input_payload,
                context=context,
                memory=memory,
                llm=llm,
                emotional_state=emotional_state,
            )
            if not isinstance(result, AgentResult):
                raise ValueError("Agent _run_impl must return AgentResult")
            run = AgentRun(
                id=new_id(),
                agent_name=self.name,
                input_payload=input_payload,
                output_payload=result.output_payload,
                status=result.status,
                emotion_delta=result.emotion_delta,
            )
            return run
        except Exception as exc:  # noqa: BLE001
            # in caso di eccezione, creiamo un AgentRun di failure
            run = AgentRun(
                id=new_id(),
                agent_name=self.name,
                input_payload=input_payload,
                output_payload={"error": str(exc)},
                status=AgentRunStatus.FAILURE,
                emotion_delta=EmotionDelta(frustration=0.1, confidence=-0.05),
            )
            return run

    @abstractmethod
    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        ...


class AgentRegistry:
    """
    Registro globale degli agent disponibili.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' già registrato")
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' non trovato")
        return self._agents[name]

    def list_agents(self) -> List[str]:
        return sorted(self._agents.keys())

# --- Registro globale + autoload da agents/ -------------------------

ACTIVE_REGISTRY: Optional[AgentRegistry] = None


def load_agents_from_packages() -> None:
    """
    Importa tutti i moduli in agents/ in modo che ciascun file
    possa registrare i propri agent nel registro globale.
    """
    import importlib
    import pkgutil
    import agents  # package agents/ nella root del progetto

    for _, module_name, _ in pkgutil.iter_modules(agents.__path__):
        importlib.import_module(f"agents.{module_name}")
