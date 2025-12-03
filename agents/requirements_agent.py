# agents/requirements_agent.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
    Message,
    MessageRole,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


def _safe_json_loads(raw: str) -> Optional[dict]:
    """
    Helper robusto: prova json.loads diretto, altrimenti prova a estrarre il primo
    blocco {...} dal testo. Se fallisce, ritorna None.
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


class RequirementsAgent(Agent):
    name = "requirements_agent"
    description = (
        "Analizza il testo dell'utente, produce una scheda formale di requisiti "
        "e la salva come memoria procedurale."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        # ------------------------------------------------------------------
        # 1) Recupero testo utente
        # ------------------------------------------------------------------
        user_message = input_payload.get("user_message", "").strip()
        if not user_message and context.messages:
            # fallback: ultimo messaggio utente nel contesto
            for msg in reversed(context.messages):
                if msg.role == MessageRole.USER:
                    user_message = msg.content
                    break

        if not user_message:
            # fallback super minimal se proprio non abbiamo testo
            text = (
                "Per poterti aiutare ho bisogno che tu mi descriva il problema, "
                "il dataset (se c'è) e cosa vuoi ottenere. Puoi scriverlo in linguaggio naturale."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": text,
                    "stop_for_user_input": True,
                    "requirements_sheet": None,
                },
                emotion_delta=EmotionDelta(curiosity=0.03),
            )

        # ------------------------------------------------------------------
        # 2) Chiamata LLM per estrarre la scheda formale
        # ------------------------------------------------------------------
        llm_input = {
            "user_request": user_message,
        }

        system_prompt = (
            "Sei un analista dei requisiti per problemi di dati e machine learning.\n"
            "Ricevi una descrizione informale del problema dell'utente e devi estrarre "
            "una scheda formale JSON con i campi principali.\n\n"
            "RESTITUISCI SOLO JSON VALIDO con la seguente struttura MINIMA:\n"
            "{\n"
            '  \"summary\": \"riassunto breve del problema in italiano\",\n'
            '  \"primary_goal\": \"obiettivo principale (es. prevedere churn, stimare vendite, esplorare dati)\",\n'
            '  \"problem_type\": \"classification|regression|clustering|time-series|exploratory|other\",\n'
            '  \"domain\": \"es. marketing, finance, operations, prodotto, altro\",\n'
            '  \"target_variable\": \"nome variabile target se nota, altrimenti null\",\n'
            '  \"input_variables\": [\"elenco\", \"variabili\", \"note\"],\n'
            '  \"dataset\": {\n'
            '    \"estimated_rows\": null,\n'
            '    \"estimated_columns\": null,\n'
            '    \"file_formats\": [\"csv\", \"parquet\", \"db\", \"altro\"],\n'
            '    \"location\": \"local|db|api|not_specified\",\n'
            '    \"db_type\": \"sqlite|postgres|mysql|bigquery|other|null\"\n'
            "  },\n"
            '  \"constraints\": {\n'
            '    \"time_budget\": \"stringa sintetica o null\",\n'
            '    \"interpretability_required\": true,\n'
            '    \"hardware_constraints\": \"stringa o null\",\n'
            '    \"tools_preferred\": [\"R\", \"Python\", \"SQL\", \"Excel\"]\n'
            "  },\n"
            '  \"evaluation\": {\n'
            '    \"metrics\": [\"accuracy\", \"auc\", \"rmse\", \"mae\", \"r2\", \"none\"],\n'
            '    \"success_criteria\": \"frase sintetica o null\"\n'
            "  },\n"
            '  \"missing_info_questions\": [\n'
            '    \"domande specifiche a cui l\\\'utente dovrebbe rispondere per chiarire i requisiti\"\n'
            "  ]\n"
            "}\n\n"
            "Regole:\n"
            "- Se un'informazione non è chiaramente deducibile, imposta il campo a null o lista vuota.\n"
            "- Le domande in missing_info_questions devono essere concrete e mirate (max 6).\n"
            "- Usa italiano semplice.\n"
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(llm_input, ensure_ascii=False),
            )
        ]

        parsed: Dict[str, Any] = {}
        try:
            raw = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=700,
            )
            parsed = _safe_json_loads(raw) or {}
        except Exception:
            parsed = {}

        # ------------------------------------------------------------------
        # 3) Costruzione scheda requisiti con default robusti
        # ------------------------------------------------------------------
        now_iso = datetime.now(timezone.utc).isoformat()
        conversation_id = getattr(context, "id", None)
        user_id = getattr(context, "user_id", None)
        project_id = getattr(context, "project_id", None) or getattr(
            context, "current_project_id", None
        )

        summary = parsed.get("summary") or user_message[:400]
        primary_goal = parsed.get("primary_goal") or "da chiarire"
        problem_type = parsed.get("problem_type") or "exploratory"
        domain = parsed.get("domain") or "unspecified"
        target_variable = parsed.get("target_variable")

        input_variables = parsed.get("input_variables") or []
        if not isinstance(input_variables, list):
            input_variables = [str(input_variables)]

        dataset_info = parsed.get("dataset") or {}
        constraints = parsed.get("constraints") or {}
        evaluation = parsed.get("evaluation") or {}

        missing_questions = parsed.get("missing_info_questions") or []
        if not isinstance(missing_questions, list):
            missing_questions = [str(missing_questions)]

        # Se l'LLM non ha proposto domande, usiamo un set di fallback
        if not missing_questions:
            missing_questions = [
                "Quante righe e colonne ha (o pensi avrà) il dataset?",
                "Qual è esattamente la variabile da prevedere o analizzare?",
                "Su quale orizzonte temporale ti interessa il risultato (es. 30 giorni, 1 anno)?",
                "Hai vincoli di tempo di calcolo o di interpretabilità del modello?",
            ]

        requirements_sheet: Dict[str, Any] = {
            "schema_version": 1,
            "created_at": now_iso,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "project_id": project_id,
            "raw_request": user_message,
            "summary": summary,
            "primary_goal": primary_goal,
            "problem_type": problem_type,
            "domain": domain,
            "target_variable": target_variable,
            "input_variables": input_variables,
            "dataset": {
                "estimated_rows": dataset_info.get("estimated_rows"),
                "estimated_columns": dataset_info.get("estimated_columns"),
                "file_formats": dataset_info.get("file_formats") or [],
                "location": dataset_info.get("location"),
                "db_type": dataset_info.get("db_type"),
            },
            "constraints": {
                "time_budget": constraints.get("time_budget"),
                "interpretability_required": constraints.get(
                    "interpretability_required"
                ),
                "hardware_constraints": constraints.get("hardware_constraints"),
                "tools_preferred": constraints.get("tools_preferred") or [],
            },
            "evaluation": {
                "metrics": evaluation.get("metrics") or [],
                "success_criteria": evaluation.get("success_criteria"),
            },
            "missing_info_questions": missing_questions,
        }

        # ------------------------------------------------------------------
        # 4) Salvataggio scheda in memoria procedurale
        # ------------------------------------------------------------------
        key_conv = (
            f"requirements_sheet:{conversation_id}"
            if conversation_id
            else "requirements_sheet"
        )
        try:
            memory.store_item(
                scope=MemoryScope.CONVERSATION,
                type_=MemoryType.PROCEDURAL,
                key=key_conv,
                content=json.dumps(requirements_sheet, ensure_ascii=False),
                metadata={"agent": self.name},
            )
        except Exception:
            pass

        # se c'è un project_id, salviamo anche a livello di progetto
        if project_id:
            key_proj = f"requirements_sheet:{project_id}"
            try:
                memory.store_item(
                    scope=MemoryScope.PROJECT,
                    type_=MemoryType.PROCEDURAL,
                    key=key_proj,
                    content=json.dumps(requirements_sheet, ensure_ascii=False),
                    metadata={"agent": self.name, "project_id": project_id},
                )
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 5) Messaggio all'utente: riassunto + domande
        # ------------------------------------------------------------------
        text = "Ho iniziato a compilare una scheda dei tuoi requisiti.\n\n"
        text += f"📌 **Riassunto del problema**:\n{summary}\n\n"
        text += f"🎯 **Obiettivo principale**: {primary_goal}\n"
        text += f"🧠 **Tipo di problema**: {problem_type}\n"
        if target_variable:
            text += f"🎯 **Target (variabile da prevedere)**: {target_variable}\n"
        if input_variables:
            text += "📎 **Variabili di input menzionate**: " + ", ".join(
                map(str, input_variables)
            ) + "\n"

        text += "\nPer completare la scheda ho ancora bisogno di alcune informazioni:\n"
        for q in missing_questions:
            text += f"- {q}\n"

        text += (
            "\nRispondi pure in modo naturale a queste domande, poi potrò usare la scheda "
            "per pianificare meglio l'analisi o la modellazione."
        )

        output = {
            "user_visible_message": text,
            "stop_for_user_input": True,  # è proprio un form di intake
            "requirements_sheet": requirements_sheet,
        }

        delta = EmotionDelta(
            curiosity=0.06,   # l'agente è interessato a chiarire il problema
            confidence=0.03,  # abbiamo una base più strutturata
        )
        return AgentResult(output_payload=output, emotion_delta=delta)
 