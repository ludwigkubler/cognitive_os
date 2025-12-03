from __future__ import annotations

import json
from typing import Any, Dict

from core.agents_base import Agent, AgentResult
from core.models import EmotionalState, EmotionDelta, MemoryKeys, MemoryScope, MemoryType, ConversationContext
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider

from r_agents.r_utils import run_r_job


class RModelingAgent(Agent):
    name = "r_modeling_agent"
    description = "Fit di un modello supervisionato in R (logistica o regressione lineare)."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        """
        input_payload atteso:
        {
          "dataset_ref": { ... come in r_eda_agent ... },
          "target": "nome_colonna_target",
          "problem_type": "classification" | "regression"
        }
        """
        dataset_ref = input_payload.get("dataset_ref")
        target = input_payload.get("target")
        problem_type = input_payload.get("problem_type", "classification")

        if not dataset_ref or not target:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "Per modellare in R ho bisogno di dataset_ref e target "
                        "(nome della colonna da predire)."
                    ),
                    "stop_for_user_input": True,
                },
                emotion_delta=EmotionDelta(curiosity=0.05),
            )

        job = {
            "analysis_type": "modeling",
            "params": {
                "dataset_ref": dataset_ref,
                "target": target,
                "problem_type": problem_type,
            },
        }

        # 1) Tentativo cache
        cached = self._try_load_cache(job, memory)
        if cached is not None:
            data = cached
        else:
            # 2) Call reale a R
            try:
                data, _ = run_r_job(
                    script_name="modeling_generic.R",
                    job=job,
                    memory=memory,
                    memory_key=MemoryKeys.R_MODELING_RESULT,
                )
            except Exception as e:
                return AgentResult(
                    output_payload={
                        "user_visible_message": (
                            "Il fitting del modello in R ha generato un errore."
                        ),
                        "error": str(e),
                        "stop_for_user_input": False,
                    },
                    emotion_delta=EmotionDelta(frustration=0.1, confidence=-0.05),
                )

        output = {
            "user_visible_message": "",
            "r_modeling_result": data,
            "stop_for_user_input": False,
        }
        delta = EmotionDelta(confidence=0.06)
        return AgentResult(output_payload=output, emotion_delta=delta)

    def _try_load_cache(
        self,
        job: Dict[str, Any],
        memory: MemoryEngine,
    ) -> Dict[str, Any] | None:
        """
        Cerca in memoria un risultato di modeling R per lo stesso job.
        Se trovato, ritorna il JSON parsato, altrimenti None.
        """
        try:
            items = memory.search_items(
                scope=MemoryScope.PROJECT,
                type_=MemoryType.PROCEDURAL,
                query=None,
                limit=50,
            )
        except Exception:
            return None

        for item in items:
            md = item.metadata or {}
            if md.get("script_name") != "modeling_generic.R":
                continue
            cached_job = md.get("job")
            if cached_job == job:
                try:
                    return json.loads(item.content)
                except Exception:
                    continue
        return None
