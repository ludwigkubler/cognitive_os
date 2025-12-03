# agents/archivist_agent.py
from __future__ import annotations

import json
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


class ArchivistAgent(Agent):
    """
    Agent che compatta e riassume memorie esistenti in un nuovo MemoryItem
    più sintetico, per evitare che la memoria esploda.
    """

    name = "archivist_agent"
    description = (
        "Riassume gruppi di memorie (per scope/tipo) in un nuovo elemento "
        "più compatto, mantenendo solo le informazioni importanti."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # -------------------------
        # 1) Parsing parametri
        # -------------------------
        scope_str = str(input_payload.get("scope", "conversation")).lower()
        scope_map = {
            "conversation": MemoryScope.CONVERSATION,
            "project": MemoryScope.PROJECT,
            "progetto": MemoryScope.PROJECT,
            "user": MemoryScope.USER,
            "utente": MemoryScope.USER,
            "global": MemoryScope.GLOBAL,
            "globale": MemoryScope.GLOBAL,
        }
        scope = scope_map.get(scope_str, MemoryScope.CONVERSATION)

        type_str = input_payload.get("type")
        type_: Optional[MemoryType] = None
        if type_str:
            t = str(type_str).lower()
            type_map = {
                "episodic": MemoryType.EPISODIC,
                "episodica": MemoryType.EPISODIC,
                "semantic": MemoryType.SEMANTIC,
                "semantica": MemoryType.SEMANTIC,
                "procedural": MemoryType.PROCEDURAL,
                "procedurale": MemoryType.PROCEDURAL,
            }
            type_ = type_map.get(t)

        query = input_payload.get("query")
        max_items_raw = input_payload.get("max_items", 50)
        try:
            max_items = int(max_items_raw)
        except Exception:
            max_items = 50
        max_items = max(1, min(max_items, 200))  # clamp tra 1 e 200

        summary_key = input_payload.get(
            "summary_key",
            f"archivist_summary_{scope.value}",
        )

        # -------------------------
        # 2) Recupero delle memorie
        # -------------------------
        items = memory.search_items(
            scope=scope,
            type_=type_,
            query=query,
            limit=max_items,
        )

        if not items:
            msg = (
                "Non ho trovato memorie da archiviare per i criteri richiesti "
                f"(scope={scope.value}, type={type_.value if type_ else 'any'}, query={query!r})."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "summary_memory_id": None,
                    "archived_item_ids": [],
                },
                emotion_delta=EmotionDelta(confidence=-0.01, frustration=0.01),
            )

        # Ordino temporalmente (dal più vecchio al più nuovo)
        sorted_items = sorted(items, key=lambda it: it.created_at)

        serializable_items: List[Dict[str, Any]] = []
        for it in sorted_items:
            serializable_items.append(
                {
                    "id": it.id,
                    "scope": it.scope.value,
                    "type": it.type.value,
                    "key": it.key,
                    "content": it.content,
                    "created_at": it.created_at.isoformat(),
                }
            )

        # -------------------------
        # 3) Prompt per il riassunto LLM
        # -------------------------
        system_prompt = (
            "Sei l'Archivist interno di un sistema multi-agent. "
            "Ricevi un elenco di memorie (log di conversazione, note, risultati di agent). "
            "Devi scrivere un riassunto compatto in italiano che mantenga solo le "
            "informazioni importanti. Non inventare fatti nuovi e non cambiare il "
            "significato. Limita il riassunto a poche frasi o pochi punti elenco brevi. "
            "Rispondi SOLO con il testo del riassunto, senza aggiungere spiegazioni meta."
        )

        user_payload = {
            "scope": scope.value,
            "type": type_.value if type_ else None,
            "query": query,
            "items": serializable_items,
        }

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(user_payload, ensure_ascii=False),
            )
        ]

        # Provo a usare l'LLM; se fallisce, fallback deterministico
        try:
            summary_text = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=512,
            )
            llm_failed = False
        except Exception as exc:  # noqa: BLE001
            joined = "\n\n".join(it.content for it in sorted_items[:5])
            summary_text = (
                "Non sono riuscito a usare il modello LLM per riassumere; "
                "qui sotto trovi un estratto delle memorie più recenti:\n\n"
                + joined
            )
            llm_failed = True

        # -------------------------
        # 4) Salvataggio del riassunto in memoria
        # -------------------------
        metadata = {
            "source_item_ids": [it.id for it in sorted_items],
            "source_scope": scope.value,
            "source_type": type_.value if type_ else None,
            "query": query,
            "num_items": len(sorted_items),
            "agent": self.name,
            "llm_used": not llm_failed,
        }

        summary_item = memory.store_item(
            scope=scope,
            type_=MemoryType.SEMANTIC,
            key=summary_key,
            content=summary_text,
            metadata=metadata,
        )

        # -------------------------
        # 5) Messaggio per l'utente
        # -------------------------
        user_msg = (
            f"Ho creato un riassunto di {len(sorted_items)} memorie "
            f"nello scope «{scope.value}» e l'ho salvato con id «{summary_item.id}» "
            f"e key «{summary_key}».\n\n"
            f"Riassunto:\n{summary_text}"
        )

        delta = EmotionDelta(
            curiosity=0.01,
            confidence=0.02,
            fatigue=0.0,
            frustration=0.0 if not llm_failed else 0.03,
        )

        return AgentResult(
            output_payload={
                "user_visible_message": user_msg,
                "summary_memory_id": summary_item.id,
                "archived_item_ids": [it.id for it in sorted_items],
            },
            emotion_delta=delta,
        )
