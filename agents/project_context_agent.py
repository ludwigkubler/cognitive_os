# agents/project_context_agent.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
    MemoryKeys,
    Message,
    MessageRole,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


def _safe_project_key(name: str) -> str:
    """
    Converte il nome progetto in una chiave "safe" per la memoria.
    es: 'Cognitive OS v1' -> 'cognitive_os_v1'
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name or "default_project"


class ProjectContextAgent(Agent):
    """
    Aggrega e aggiorna il contesto di un progetto:

    - legge memorie PROJECT esistenti,
    - guarda gli ultimi messaggi di conversazione,
    - opzionalmente usa note extra e preview di file,
    - genera un riassunto strutturato tramite LLM,
    - salva il risultato in memoria PROJECT come SEMANTIC.

    Input tipico (input_payload):
    {
        "project_name": "cognitive_os",
        "mode": "update",  # o "summarize"
        "extra_notes": "sto lavorando sugli agent R...",
        "files": ["/path/a/un/file.py", ...]  # opzionale, best-effort
    }
    """

    name = "project_context_agent"
    description = (
        "Costruisce/aggiorna il contesto di un progetto (riassunto, TODO, rischi) "
        "e lo salva in memoria PROJECT."
    )

    # ------------------------------------------------------------------ #
    # Entry point logico
    # ------------------------------------------------------------------ #
    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        project_name = (
            input_payload.get("project_name")
            or input_payload.get("name")
            or "default_project"
        )
        project_name = str(project_name).strip() or "default_project"
        project_key = _safe_project_key(project_name)

        mode = input_payload.get("mode", "update")
        extra_notes = input_payload.get("extra_notes") or ""
        files: List[str] = input_payload.get("files") or []
        max_memories: int = int(input_payload.get("max_memories", 15))
        max_recent_msgs: int = int(input_payload.get("max_recent_messages", 12))

        # 1) Prendo memorie PROJECT correlate
        existing_items = memory.search_items(
            scope=MemoryScope.PROJECT,
            type_=None,
            query=project_key,
            limit=max_memories,
        )

        existing_for_llm = [
            {
                "id": m.id,
                "key": m.key,
                "content": m.content[:600],  # limito la lunghezza
                "metadata": m.metadata,
            }
            for m in existing_items
        ]

        # 2) Snippet di conversazione recente
        recent_messages: List[Message] = memory.get_recent_messages(
            limit=max_recent_msgs
        )
        conv_snippet: List[Dict[str, Any]] = []
        for msg in recent_messages:
            conv_snippet.append(
                {
                    "role": msg.role.value,
                    "content": msg.content[:400],
                }
            )

        # 3) Preview file opzionali (best effort)
        files_preview: List[Dict[str, str]] = []
        for path in files[:5]:  # non più di 5 file
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(4000)
                files_preview.append(
                    {
                        "path": path,
                        "content_snippet": content,
                    }
                )
            except Exception:
                files_preview.append(
                    {
                        "path": path,
                        "content_snippet": "<errore nella lettura del file>",
                    }
                )

        # 4) Costruisco payload per LLM
        llm_input = {
            "project_name": project_name,
            "project_key": project_key,
            "mode": mode,
            "extra_notes": extra_notes,
            "conversation_snippet": conv_snippet,
            "existing_project_memories": existing_for_llm,
            "files_preview": files_preview,
        }

        # 5) Prompt LLM: chiedo un riassunto strutturato in ITA
        system_prompt = (
            "Sei il ProjectContextAgent di un sistema cognitivo multi-agent.\n"
            "Ricevi in formato JSON informazioni su un progetto (nome, snippet di "
            "conversazione recente, memorie precedenti, eventuali note e snippet di file).\n\n"
            "Devi produrre UNA SINTESI TESTUALE in italiano, ben strutturata, con questa forma:\n\n"
            "Nome progetto: ...\n"
            "Obiettivo principale:\n"
            "- ...\n\n"
            "Stato attuale:\n"
            "- ...\n\n"
            "Cose già fatte:\n"
            "- ...\n\n"
            "TODO a breve:\n"
            "- ...\n\n"
            "Rischi / problemi aperti:\n"
            "- ...\n\n"
            "Note personali del sistema:\n"
            "- ...\n\n"
            "Linee guida importanti:\n"
            "- NON aggiungere spiegazioni fuori da questa struttura.\n"
            "- NON parlare di te stesso come LLM, parla come sistema che organizza il progetto.\n"
        )

        summary_text: str

        if llm is not None:
            try:
                from core.models import Message, MessageRole  # type: ignore

                messages = [
                    Message(
                        role=MessageRole.USER,
                        content=json.dumps(llm_input, ensure_ascii=False),
                    )
                ]
                raw = llm.generate(
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=900,
                )
                summary_text = raw.strip()
            except Exception as exc:  # fallback deterministico
                summary_text = self._fallback_summary(
                    project_name=project_name,
                    existing_items=existing_items,
                    extra_notes=extra_notes,
                    error=str(exc),
                )
        else:
            # nessun LLM reale → fallback deterministico
            summary_text = self._fallback_summary(
                project_name=project_name,
                existing_items=existing_items,
                extra_notes=extra_notes,
            )

        # 6) Scrivo in memoria PROJECT
        metadata = {
            "project_name": project_name,
            "project_key": project_key,
            "mode": mode,
            "source": self.name,
            "num_existing_items": len(existing_items),
            "files": files,
        }

        item = memory.store_item(
            scope=MemoryScope.PROJECT,
            type_=MemoryType.SEMANTIC,
            key=f"{MemoryKeys.PROJECT_CONTEXT_PREFIX}{project_key}",
            content=summary_text,
            metadata=metadata,
        )

        user_msg = (
            f"Ho aggiornato il contesto per il progetto «{project_name}» "
            f"(chiave: {project_key}).\n\n"
            f"Ecco un riassunto:\n\n{summary_text}"
        )

        output = {
            "user_visible_message": user_msg,
            "project_name": project_name,
            "project_key": project_key,
            "stored_item_id": item.id,
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(
            confidence=0.04,
            curiosity=-0.01,  # leggero calo dopo lavoro organizzativo
            fatigue=0.02,
        )
        return AgentResult(output_payload=output, emotion_delta=delta)

    # ------------------------------------------------------------------ #
    # Fallback deterministico se l'LLM non è disponibile o va in errore
    # ------------------------------------------------------------------ #
    def _fallback_summary(
        self,
        project_name: str,
        existing_items: List[Any],
        extra_notes: str,
        error: Optional[str] = None,
    ) -> str:
        lines = []
        lines.append(f"Nome progetto: {project_name}")
        lines.append("Obiettivo principale:")
        lines.append("- (non definito: servirebbe un LLM per inferirlo)")

        lines.append("\nStato attuale:")
        if existing_items:
            lines.append(
                f"- Sono presenti {len(existing_items)} memorie PROJECT correlate."
            )
        else:
            lines.append("- Nessuna memoria PROJECT trovata per questo progetto.")

        if extra_notes:
            lines.append("\nNote recenti:")
            lines.append(f"- {extra_notes}")

        if error:
            lines.append("\n[Nota tecnica]")
            lines.append(f"- Riassunto generato senza LLM a causa di: {error}")

        lines.append("\nTODO a breve:")
        lines.append("- Definire obiettivi e milestone più dettagliate.")

        lines.append("\nRischi / problemi aperti:")
        lines.append("- Mancanza di un riassunto LLM accurato del contesto.")
        return "\n".join(lines)