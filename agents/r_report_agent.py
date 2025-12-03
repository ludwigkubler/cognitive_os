from __future__ import annotations

from typing import Any, Dict

from core.agents_base import Agent, AgentResult
from core.models import EmotionalState, EmotionDelta, MemoryKeys, ConversationContext
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider

from r_agents.r_utils import run_r_job


class RReportAgent(Agent):
    name = "r_report_agent"
    description = "Compone un piccolo report statistico in R a partire dai risultati EDA + modeling."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        """
        input_payload atteso (tutto opzionale, prende da memoria se non c'è):
        {
          "eda_result": {...},
          "modeling_result": {...}
        }
        """
        eda_result = input_payload.get("eda_result")
        modeling_result = input_payload.get("modeling_result")

        # fallback: prova a recuperare dalla memoria
        if eda_result is None:
            try:
                eda_result = memory.load_item_content(
                    key="r_eda_result"
                )
            except Exception:
                pass
        if modeling_result is None:
            try:
                modeling_result = memory.load_item_content(
                    key="r_modeling_result"
                )
            except Exception:
                pass

        job = {
            "analysis_type": "report",
            "params": {
                "eda_result": eda_result,
                "modeling_result": modeling_result,
            },
        }

        try:
            data, _ = run_r_job(
                script_name="report_generic.R",
                job=job,
                memory=memory,
                memory_key=MemoryKeys.R_REPORT_RESULT,
            )
        except Exception as e:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "La generazione del report in R ha dato errore."
                    ),
                    "error": str(e),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.06),
            )

        output = {
            "user_visible_message": "",
            "r_report_result": data,
            "stop_for_user_input": False,
        }
        delta = EmotionDelta(confidence=0.04)
        return AgentResult(output_payload=output, emotion_delta=delta)
