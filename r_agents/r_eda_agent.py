from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from core.agents_base import Agent, AgentResult
from core.models import EmotionalState, EmotionDelta, ConversationContext, MemoryScope, MemoryType
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider

from .r_utils import run_r_job, RJobError


class REdaAgent(Agent):
    name = "r_eda_agent"
    description = "Esegue una EDA generica in R su un dataset (SQLite o CSV)."

    def _validate_dataset_ref(self, dataset_ref: Any) -> Tuple[bool, str]:
        if not isinstance(dataset_ref, dict):
            return False, "dataset_ref deve essere un oggetto/dizionario."

        dtype = dataset_ref.get("type")
        if dtype not in {"sqlite_table", "csv"}:
            return False, "dataset_ref.type deve essere 'sqlite_table' oppure 'csv'."

        path = dataset_ref.get("path")
        if not isinstance(path, str) or not path.strip():
            return False, "dataset_ref.path deve essere il path a .db o .csv."

        if dtype == "sqlite_table":
            table = dataset_ref.get("table")
            if not isinstance(table, str) or not table.strip():
                return False, "dataset_ref.table è obbligatorio per type='sqlite_table'."

        return True, ""

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        dataset_ref = input_payload.get("dataset_ref")
        ok, err_msg = self._validate_dataset_ref(dataset_ref)

        if not ok:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "Per eseguire la EDA in R ho bisogno di un `dataset_ref` valido.\n"
                        f"Dettaglio: {err_msg}"
                    ),
                    "expected_input_example": {
                        "dataset_ref": {
                            "type": "sqlite_table",
                            "path": "/percorso/al/file.db",
                            "table": "nome_tabella",
                        }
                    },
                    "stop_for_user_input": True,
                },
                emotion_delta=EmotionDelta(curiosity=0.03, frustration=0.02),
            )

        job: Dict[str, Any] = {
            "analysis_type": "eda",
            "params": {
                "dataset_ref": dataset_ref,
            },
        }

        # 1) Tentativo di usare la cache
        cached = self._try_load_cache(job, memory)
        if cached is not None:
            data = cached
        else:
            # 2) Altrimenti eseguo davvero R
            try:
                data, _ = run_r_job(
                    script_name="eda_generic.R",
                    job=job,
                    memory=memory,
                    memory_key="r_eda_result",
                    scope=MemoryScope.PROJECT,
                    type_=MemoryType.PROCEDURAL,
                )
            except RJobError as e:
                return AgentResult(
                    output_payload={
                        "user_visible_message": (
                            "La EDA in R ha generato un errore interno.\n"
                            "Controlla i log o l'output di R per i dettagli tecnici."
                        ),
                        "error": str(e),
                        "stop_for_user_input": False,
                    },
                    emotion_delta=EmotionDelta(frustration=0.08, confidence=-0.05),
                )

        output = {
            "user_visible_message": "",  # ExplanationAgent la racconterà
            "r_eda_result": data,
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(confidence=0.04, curiosity=0.03)
        return AgentResult(output_payload=output, emotion_delta=delta)

    def _try_load_cache(
        self,
        job: Dict[str, Any],
        memory: MemoryEngine,
    ) -> Dict[str, Any] | None:
        """
        Cerca in memoria un risultato EDA R già calcolato per lo stesso job.
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
            if md.get("script_name") != "eda_generic.R":
                continue
            cached_job = md.get("job")
            if cached_job == job:
                try:
                    return json.loads(item.content)
                except Exception:
                    continue
        return None
