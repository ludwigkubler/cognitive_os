from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
    AgentRunStatus,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class RAnalysisAgent(Agent):
    name = "r_analysis_agent"
    description = "Bridge generico verso script R parametrizzati (per ora: churn_demo)."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        analysis_type = input_payload.get("analysis_type", "churn_demo")
        params = input_payload.get("params", {}) or {}

        project_root = Path(__file__).resolve().parents[1]
        r_dir = project_root / "r_agents"

        if analysis_type == "churn_demo":
            script = r_dir / "churn_analysis.R"
        else:
            # in futuro: mappare altri tipi → script diversi
            script = r_dir / "churn_analysis.R"

        job = {
            "analysis_type": analysis_type,
            "params": params,
        }

        try:
            cmd = ["Rscript", str(script), json.dumps(job)]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "Non riesco a trovare 'Rscript' nel PATH. "
                        "Installa R oppure configura correttamente l'ambiente."
                    ),
                    "error": str(e),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.1, confidence=-0.05),
                status=AgentRunStatus.FAILURE,
            )

        if proc.returncode != 0:
            err = proc.stderr.strip() or f"Exit code {proc.returncode}"
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "Lo script R ha restituito un errore di esecuzione. "
                        "Vedi dettagli in log interno."
                    ),
                    "error": err,
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.1, confidence=-0.05),
                status=AgentRunStatus.FAILURE,
            )

        stdout = proc.stdout.strip()
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "Lo script R ha prodotto output non-JSON. "
                        "Non riesco a interpretarlo."
                    ),
                    "raw_output": stdout[:1000],
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.08, confidence=-0.05),
                status=AgentRunStatus.FAILURE,
            )

        # nuovo: controllo protocollo logico ok/error dallo script R
        if isinstance(data, dict) and data.get("ok") is False:
            err_msg = data.get("error") or "Errore logico riportato dallo script R."
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "L'analisi di churn in R non è andata a buon fine "
                        "(errore logico interno)."
                    ),
                    "error": err_msg,
                    "r_result": data,  # salvo comunque il payload, può essere utile per debug
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.08, confidence=-0.05),
                status=AgentRunStatus.FAILURE,
            )

        # Salviamo in memoria procedurale
        try:
            memory.store_item(
                scope=MemoryScope.CONVERSATION,
                type_=MemoryType.PROCEDURAL,
                key=f"r_result_{analysis_type}",
                content=json.dumps(data),
            )
        except Exception:
            pass

        output = {
            "user_visible_message": "",  # ExplanationAgent gestisce la parte narrativa
            "r_result": data,
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(confidence=0.05, curiosity=0.03)
        return AgentResult(output_payload=output, emotion_delta=delta)
