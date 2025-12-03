from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
    MessageRole,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class MemoryAgent(Agent):
    """
    Agente di memoria esplicita + etichettatura per il profilo utente.

    Scopo principale:
      - permettere all'utente di dire “ricordati che…” / “salva in memoria…”
        e creare un MemoryItem nella memoria persistente.

    Estensione per il futuro sistema relazionale:
      - quando scope=USER e type=SEMANTIC, marca la memoria come candidata
        all'aggiornamento del profilo utente (ProfileUpdateAgent / PreferenceLearnerAgent).
      - supporta campi come 'mode' e 'category' per distinguere:
          * 'note'        → nota generica
          * 'preference'  → gusti / cose che piacciono o non piacciono
          * 'hobby'       → interessi personali
          * 'fact'        → fatto sul mondo o sulla persona
          * 'teaching'    → qualcosa che l'utente insegna al sistema

    input_payload previsto (tutto opzionale, se il Router non lo prepara):

      {
        "content": "testo da memorizzare",
        "scope": "user" | "project" | "global" | "conversation",
        "type": "semantic" | "episodic" | "procedural",
        "key": "chiave_logica",
        "tags": ["lista", "di", "tag"],
        "importance": 0.0-1.0 (float),
        "mode": "note" | "preference" | "hobby" | "fact" | "teaching",
        "category": "topic opzionale es: calcio, musica"
      }

    Se content non è presente, prova a usare l'ultimo messaggio USER
    della conversazione. Se ancora nulla, chiede esplicitamente all'utente.
    """

    name = "memory_agent"
    description = (
        "Memorizza in modo esplicito informazioni importanti su richiesta "
        "dell'utente (note, preferenze, fatti) e le etichetta per il profilo utente."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        # 1) Decidiamo cosa memorizzare
        content, from_last_user = self._resolve_content(input_payload, context)

        if not content or not content.strip():
            # Non sappiamo cosa memorizzare → chiediamo chiarimento
            msg = (
                "Dimmi esattamente che cosa vuoi che memorizzi. "
                "Per esempio: «ricordati che preferisco usare Ubuntu per lo sviluppo»."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": True,
                },
                emotion_delta=EmotionDelta(curiosity=0.03),
            )

        content = content.strip()

        # 2) Scope: dove deve vivere questa memoria?
        scope = self._resolve_scope(input_payload)

        # 3) Tipo di memoria: semantica/episodica/procedurale
        mem_type = self._resolve_type(input_payload)

        # 4) Key logica
        key = self._resolve_key(input_payload, content)

        # 5) Tags / importanza / mode / category
        tags = self._resolve_tags(input_payload)
        importance = self._resolve_importance(input_payload)
        mode = self._resolve_mode(input_payload)
        category = self._resolve_category(input_payload)

        user_id = getattr(context, "user_id", None)

        metadata: Dict[str, Any] = {
            "tags": tags,
            "importance": importance,
            "from_last_user_message": from_last_user,
            "agent": self.name,
            "mode": mode,
            "category": category,
            "user_id": user_id,
        }

        # 6) Marcare come candidata al profilo utente (se appropriato)
        #    Questo NON aggiorna direttamente il profilo,
        #    ma aiuta ProfileUpdateAgent / PreferenceLearnerAgent a selezionare cosa usare.
        if scope == MemoryScope.USER and mem_type == MemoryType.SEMANTIC:
            metadata["profile_candidate"] = True

            # se l'utente ha specificato mode "preference" o "hobby" o "teaching",
            # aggiungiamo tag coerenti per facilitare il retrieval.
            if mode in {"preference", "hobby", "teaching"}:
                extra_tags = [mode]
                if category:
                    extra_tags.append(category)
                metadata.setdefault("tags", [])
                for t in extra_tags:
                    if t not in metadata["tags"]:
                        metadata["tags"].append(t)

        # 7) Salvataggio su SQLite tramite MemoryEngine
        item = memory.store_item(
            scope=scope,
            type_=mem_type,
            key=key,
            content=content,
            metadata=metadata,
        )

        # 8) Messaggio di conferma per l'utente
        user_msg_lines: List[str] = []
        user_msg_lines.append("Ok, lo memorizzo.")
        user_msg_lines.append(
            f"- scope: {scope.value}, type: {mem_type.value}, key: «{key}»"
        )
        if mode:
            user_msg_lines.append(f"- modalità: {mode}")
        if category:
            user_msg_lines.append(f"- categoria/argomento: {category}")
        if tags:
            user_msg_lines.append(f"- tags: {', '.join(tags)}")
        user_msg_lines.append(f"- id interno: {item.id}")

        delta = EmotionDelta(
            confidence=0.03,
            curiosity=0.01,
        )

        return AgentResult(
            output_payload={
                "user_visible_message": "\n".join(user_msg_lines),
                "stored_memory_id": item.id,
                "scope": scope.value,
                "type": mem_type.value,
                "key": key,
                "mode": mode,
                "category": category,
                "stop_for_user_input": False,
            },
            emotion_delta=delta,
        )

    # ------------------------------------------------------------------
    # Helpers di risoluzione parametri
    # ------------------------------------------------------------------

    def _resolve_content(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
    ) -> tuple[Optional[str], bool]:
        """
        Determina il testo da memorizzare.
        Ritorna (content, from_last_user_message).
        """
        # preferiamo content esplicito
        for k in ("content", "text", "note", "user_message"):
            if k in input_payload and isinstance(input_payload[k], str):
                c = input_payload[k].strip()
                if c:
                    return c, False

        # altrimenti usiamo l'ultimo messaggio USER
        for msg in reversed(context.messages):
            if msg.role == MessageRole.USER:
                return msg.content, True

        return None, False

    def _resolve_scope(self, input_payload: Dict[str, Any]) -> MemoryScope:
        raw = str(input_payload.get("scope", "user")).lower()

        if raw in {"user", "utente"}:
            return MemoryScope.USER
        if raw in {"project", "progetto"}:
            return MemoryScope.PROJECT
        if raw in {"global", "globale", "system"}:
            return MemoryScope.GLOBAL
        if raw in {"conversation", "conversazione", "sessione"}:
            return MemoryScope.CONVERSATION

        # default sicuro: preferenze dell'utente
        return MemoryScope.USER

    def _resolve_type(self, input_payload: Dict[str, Any]) -> MemoryType:
        raw = str(input_payload.get("type", "semantic")).lower()

        if raw in {"episodic", "episodica"}:
            return MemoryType.EPISODIC
        if raw in {"procedural", "procedurale"}:
            return MemoryType.PROCEDURAL
        # default: semantica
        return MemoryType.SEMANTIC

    def _resolve_key(self, input_payload: Dict[str, Any], content: str) -> str:
        if "key" in input_payload and isinstance(input_payload["key"], str):
            k = input_payload["key"].strip()
            if k:
                return k

        # generiamo una chiave leggera dal contenuto
        # prendiamo le prime 5 parole alfanumeriche
        tokens = re.findall(r"[a-zA-Z0-9]+", content.lower())
        if not tokens:
            return "note"
        key = "_".join(tokens[:5])
        return key[:80]  # limitiamo la lunghezza

    def _resolve_tags(self, input_payload: Dict[str, Any]) -> List[str]:
        tags_raw = input_payload.get("tags")
        if tags_raw is None:
            return []

        if isinstance(tags_raw, str):
            # separiamo per virgola o spazio
            parts = re.split(r"[,\s]+", tags_raw)
            return [p for p in (x.strip() for x in parts) if p]

        if isinstance(tags_raw, list):
            out: List[str] = []
            for t in tags_raw:
                if isinstance(t, str):
                    tt = t.strip()
                    if tt:
                        out.append(tt)
            return out

        return []

    def _resolve_importance(self, input_payload: Dict[str, Any]) -> float:
        raw = input_payload.get("importance")
        if raw is None:
            return 0.5
        try:
            v = float(raw)
        except Exception:
            return 0.5
        # clamp tra 0 e 1
        return max(0.0, min(1.0, v))

    def _resolve_mode(self, input_payload: Dict[str, Any]) -> str:
        raw = str(input_payload.get("mode", "")).strip().lower()
        if raw in {"note", "preference", "hobby", "fact", "teaching"}:
            return raw
        return ""  # nessuna modalità esplicita

    def _resolve_category(self, input_payload: Dict[str, Any]) -> str:
        cat = input_payload.get("category")
        if isinstance(cat, str):
            cat = cat.strip()
            return cat
        return ""
