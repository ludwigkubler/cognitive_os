# agents/user_profile_agent.py
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
    Tenta di parsare JSON in modo tollerante:
    - prima prova json.loads(raw)
    - se fallisce, prova a estrarre la sottostringa tra la prima '{'
      e l'ultima '}' e a fare json.loads su quella.

    Ritorna un dict se ci riesce, altrimenti None.
    """
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # fallback: estrai contenuto tra la prima '{' e l'ultima '}'
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

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_base_profile(user_id: str, raw_profile: Optional[str]) -> Dict[str, Any]:
    """
    Prende il contenuto JSON grezzo del profilo (o None) e ritorna
    un dict conforme allo schema di base, con default sensati.
    """
    if raw_profile:
        try:
            data = json.loads(raw_profile)
            if isinstance(data, dict):
                # versione minima: se manca schema_version/user_id, li aggiungiamo
                data.setdefault("schema_version", 1)
                data.setdefault("user_id", user_id)
                data.setdefault("meta", {})
                data["meta"].setdefault("last_profile_update", _utc_now_iso())
                return data
        except Exception:
            # se il JSON è corrotto, ripartiamo da zero
            pass

    # Profilo nuovo di default
    now = _utc_now_iso()
    return {
        "schema_version": 1,
        "user_id": user_id,
        "display_name": user_id,
        "last_seen": now,
        "basic_info": {
            "age_range": None,
            "location": None,
            "preferred_language": "it",
        },
        "interaction_style": {
            "prefers_short_answers": False,
            "likes_technical_detail": True,
            "likes_humor": True,
            "sensitivity_level": "medium",
            "formality": "casual",
        },
        "topics": {},
        "avoid_topics": [],
        "hobbies": [],
        "values": [],
        "conversational_prefs": {
            "likes_deep_conversations": True,
            "likes_current_events": True,
            "avoid_politics": "maybe",
            "privacy_boundaries": "",
            "comfortable_with_personal_questions": "medium",
        },
        "recent_themes": [],
        "open_questions": [],
        "relationship_with_system": {
            "trust_level": 0.5,
            "comfort_level": 0.5,
            "notes": "",
        },
        "conversation_stats": {
            "total_sessions": 0,
            "total_messages": 0,
            "first_seen": now,
            "last_session_summary_id": None,
        },
        "meta": {
            "last_profile_update": now,
            "updated_by_agent": "user_profile_agent",
            "notes": "",
        },
    }


class UserProfileAgent(Agent):
    """
    ProfileUpdateAgent / UserModelAgent

    Scopo:
      - Leggere le ultime interazioni (messaggi + memorie utente),
      - Leggere il profilo utente attuale (se esiste),
      - Chiedere all'LLM di aggiornare il profilo secondo lo schema condiviso,
      - Salvare il nuovo profilo in memoria:
          scope = USER
          type  = SEMANTIC
          key   = f"user_profile:{user_id}"

    Input atteso (input_payload):
      {
        "user_id": "user-1",              # opzionale, default context.user_id
        "max_messages": 30,               # opzionale
        "max_user_memories": 50           # opzionale
      }

    Output:
      - user_visible_message: breve riassunto di cosa è stato appreso
      - learned_facts: lista di stringhe (fatti appresi/aggiornati)
      - profile_memory_id: id del MemoryItem salvato
    """

    name = "user_profile_agent"
    description = (
        "Aggiorna il profilo utente (preferenze, hobby, temi, valori) "
        "in base alla conversazione recente e alle memorie utente."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # -------------------------------------------------------------
        # 1) Identifica l'utente e parametri base
        # -------------------------------------------------------------
        user_id = input_payload.get("user_id") or getattr(context, "user_id", None)
        if not user_id:
            msg = (
                "UserProfileAgent: non riesco a determinare lo user_id. "
                "Serve context.user_id o input_payload['user_id']."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                    "learned_facts": [],
                    "profile_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.02, confidence=-0.02),
            )

        max_messages = int(input_payload.get("max_messages", 30))
        max_user_memories = int(input_payload.get("max_user_memories", 50))

        # clamp semplice
        max_messages = max(5, min(max_messages, 200))
        max_user_memories = max(10, min(max_user_memories, 200))

        # -------------------------------------------------------------
        # 2) Recupera profilo attuale (se esiste)
        # -------------------------------------------------------------
        profile_key = f"user_profile:{user_id}"
        raw_profile = memory.load_item_content(
            key=profile_key,
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
        )
        base_profile = _ensure_base_profile(user_id=user_id, raw_profile=raw_profile)

        # -------------------------------------------------------------
        # 3) Costruisce input per LLM: messaggi + memorie utente
        # -------------------------------------------------------------
        # a) Ultimi N messaggi della conversazione
        #    (se il context è lungo, tagliamo a coda)
        recent_msgs: List[Message] = context.messages[-max_messages:]

        serializable_messages: List[Dict[str, Any]] = []
        for m in recent_msgs:
            serializable_messages.append(
                {
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
            )

        # b) Memorie utente (scope=USER, type=SEMANTIC)
        user_memories = memory.search_items(
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
            query=None,
            limit=max_user_memories,
        )

        serializable_memories: List[Dict[str, Any]] = []
        for it in user_memories:
            serializable_memories.append(
                {
                    "id": it.id,
                    "key": it.key,
                    "content": it.content,
                    "metadata": it.metadata,
                    "created_at": it.created_at.isoformat(),
                }
            )

        # -------------------------------------------------------------
        # 4) Prompt all'LLM: aggiorna il profilo secondo lo schema
        # -------------------------------------------------------------
        system_prompt = (
            "Sei l'UserProfileAgent di un sistema cognitivo multi-agent. "
            "Il tuo compito è aggiornare un profilo utente strutturato in JSON "
            "in base alla conversazione recente e alle memorie disponibili.\n\n"
            "REQUISITI IMPORTANTI:\n"
            "- Mantieni lo schema del profilo esistente (campi principali invariati).\n"
            "- Aggiorna solo ciò che è supportato dalle evidenze (messaggi/memorie).\n"
            "- Non inventare fatti non supportati.\n"
            "- Se una preferenza è espressa con chiarezza (es. 'odio il calcio'), "
            "  aggiorna topics e avoid_topics di conseguenza.\n"
            "- Non cancellare informazioni utili già presenti nel profilo: "
            "  se non c'è conflitto, mantieni e arricchisci.\n"
            "- Aggiorna last_seen e conversation_stats (total_messages, ecc.).\n"
            "- Usa SEMPRE la lingua italiana nei testi (notes, values, ecc.).\n\n"
            "DEVI RISPONDERE SOLO CON UN JSON della forma:\n"
            "{\n"
            "  \\\"updated_profile\\\": { \"__PROFILO_COMPLETO__\" },\\n"
            '  \"learned_facts\": [\"stringa\", ...]\n'
            "}\n"
        )

        llm_input = {
            "user_id": user_id,
            "current_profile": base_profile,
            "recent_messages": serializable_messages,
            "user_memories": serializable_memories,
        }

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(llm_input, ensure_ascii=False),
            )
        ]

        try:
            llm_raw = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                "UserProfileAgent: errore durante la chiamata all'LLM per aggiornare il profilo. "
                f"Dettagli: {exc}"
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                    "learned_facts": [],
                    "profile_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.1, confidence=-0.05),
            )

        # -------------------------------------------------------------
        # 5) Parse dell'output LLM (JSON)
        # -------------------------------------------------------------
        learned_facts: List[str] = []
        updated_profile = base_profile

        learned_facts: List[str] = []
        updated_profile = base_profile

        parsed = _safe_json_loads(llm_raw)
        if parsed is None:
            learned_facts = ["Impossibile parsare JSON dall'LLM; profilo lasciato invariato."]
        else:
            maybe_profile = parsed.get("updated_profile")
            if isinstance(maybe_profile, dict):
                updated_profile = maybe_profile
            lf = parsed.get("learned_facts") or []
            if isinstance(lf, list):
                learned_facts = [str(x) for x in lf]

        # Aggiornamento minimo di sicurezza su meta
        meta = updated_profile.setdefault("meta", {})
        meta["last_profile_update"] = _utc_now_iso()
        meta["updated_by_agent"] = "user_profile_agent"

        # -------------------------------------------------------------
        # 6) Salvataggio in memoria persistente
        # -------------------------------------------------------------
        try:
            item = memory.store_item(
                scope=MemoryScope.USER,
                type_=MemoryType.SEMANTIC,
                key=profile_key,
                content=json.dumps(updated_profile, ensure_ascii=False),
                metadata={
                    "agent": self.name,
                    "user_id": user_id,
                    "schema_version": updated_profile.get("schema_version", 1),
                    "learned_facts": learned_facts,
                },
            )
            profile_memory_id = item.id
        except Exception as exc:  # noqa: BLE001
            profile_memory_id = None
            learned_facts.append(f"Errore nel salvataggio del profilo: {exc}")

        # -------------------------------------------------------------
        # 7) Messaggio sintetico per l'utente
        # -------------------------------------------------------------
        lines: List[str] = [f"Ho aggiornato il tuo profilo interno (utente: {user_id})."]

        if learned_facts:
            lines.append("")
            lines.append("Nuovi fatti appresi / aggiornati:")
            for f in learned_facts[:8]:
                lines.append(f"- {f}")

            if len(learned_facts) > 8:
                lines.append(f"... e altri {len(learned_facts) - 8} elementi.")

        if profile_memory_id:
            lines.append("")
            lines.append(f"(Profilo salvato in memoria interna con id: {profile_memory_id}.)")

        output = {
            "user_visible_message": "\n".join(lines),
            "stop_for_user_input": False,
            "user_id": user_id,
            "profile_memory_id": profile_memory_id,
            "learned_facts": learned_facts,
        }

        delta = EmotionDelta(
            curiosity=0.03,
            confidence=0.03,
            fatigue=0.01,
        )

        return AgentResult(output_payload=output, emotion_delta=delta)
