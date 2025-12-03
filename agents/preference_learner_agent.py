# agents/preference_learner_agent.py
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

# Riutilizziamo lo stesso helper usato da UserProfileAgent
# per garantire coerenza di schema nel profilo utente.
try:
    from agents.user_profile_agent import _ensure_base_profile
except ImportError:
    # Fallback minimale se per qualche motivo non riusciamo a importarlo.
    def _ensure_base_profile(user_id: str, raw_profile: Optional[str]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        if raw_profile:
            try:
                data = json.loads(raw_profile)
                if isinstance(data, dict):
                    data.setdefault("schema_version", 1)
                    data.setdefault("user_id", user_id)
                    data.setdefault("meta", {})
                    data["meta"].setdefault("last_profile_update", now)
                    return data
            except Exception:
                pass

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
                "updated_by_agent": "preference_learner_agent",
                "notes": "",
            },
        }

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


class PreferenceLearnerAgent(Agent):
    """
    Agent specializzato nell'apprendimento di PREFERENZE dall'utente.

    Scopo:
      - Riconoscere espressioni del tipo:
          * "non parliamo di calcio che mi annoia"
          * "adoro parlare di musica"
          * "odio quando mi chiedono di lavoro"
      - Estrarre da conversazione + memorie esplicite frasi di preferenza/avversione.
      - Aggiornare il profilo utente SOLO nelle sezioni:
          * topics (like/avoid per argomenti specifici),
          * avoid_topics,
          * hobbies,
          * conversational_prefs.

    Input atteso (input_payload):
      {
        "user_id": "user-1",            # opzionale, default context.user_id
        "max_messages": 40,             # opzionale, finestra conversazione
        "max_preference_memories": 50   # opzionale, memorie con mode=preference/hobby
      }

    Output:
      - user_visible_message: breve riepilogo delle preferenze apprese/aggiornate
      - preference_updates: lista strutturata di aggiornamenti effettuati
      - profile_memory_id: id del MemoryItem del profilo aggiornato (se salvato)
    """

    name = "preference_learner_agent"
    description = (
        "Analizza la conversazione e le memorie utente per aggiornare "
        "preferenze, argomenti da evitare, hobby e stile conversazionale "
        "all'interno del profilo utente."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        # -------------------------------------------------------------
        # 1) Identifica l'utente e parametri base
        # -------------------------------------------------------------
        user_id = input_payload.get("user_id") or getattr(context, "user_id", None)
        if not user_id:
            msg = (
                "PreferenceLearnerAgent: non riesco a determinare lo user_id. "
                "Serve context.user_id o input_payload['user_id']."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                    "preference_updates": [],
                    "profile_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.02, confidence=-0.02),
            )

        max_messages = int(input_payload.get("max_messages", 40))
        max_pref_mems = int(input_payload.get("max_preference_memories", 50))

        max_messages = max(5, min(max_messages, 200))
        max_pref_mems = max(10, min(max_pref_mems, 200))

        # -------------------------------------------------------------
        # 2) Carica profilo attuale
        # -------------------------------------------------------------
        profile_key = f"user_profile:{user_id}"
        raw_profile = memory.load_item_content(
            key=profile_key,
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
        )
        base_profile = _ensure_base_profile(user_id=user_id, raw_profile=raw_profile)

        # -------------------------------------------------------------
        # 3) Costruisce input per l'LLM
        #    - finestre di conversazione recenti
        #    - memorie marcate come preferenze/hobby/teaching
        # -------------------------------------------------------------
        # a) Ultimi N messaggi della conversazione
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

        # b) Memorie utente candidate (scope=USER, type=SEMANTIC, profile_candidate o tag)
        all_user_semantic = memory.search_items(
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
            query=None,
            limit=max_pref_mems,
        )

        candidate_memories: List[Dict[str, Any]] = []
        for it in all_user_semantic:
            md = it.metadata or {}
            tags = md.get("tags", []) or []
            if not isinstance(tags, list):
                tags = []
            tags_norm = [str(t).lower() for t in tags]

            profile_candidate = bool(md.get("profile_candidate", False))
            mode = str(md.get("mode", "")).lower()
            category = md.get("category")

            # Se è marcata come candidata profilo, o ha mode/ tag coerenti, la usiamo
            if profile_candidate or mode in {"preference", "hobby", "teaching"} or any(
                t in {"preference", "hobby"} for t in tags_norm
            ):
                candidate_memories.append(
                    {
                        "id": it.id,
                        "key": it.key,
                        "content": it.content,
                        "metadata": md,
                        "created_at": it.created_at.isoformat(),
                    }
                )

        # -------------------------------------------------------------
        # 4) Prompt all'LLM focalizzato sulle PREFERENZE
        # -------------------------------------------------------------
        system_prompt = (
            "Sei il PreferenceLearnerAgent di un sistema cognitivo multi-agent.\n"
            "Il tuo compito è aggiornare SOLO le parti del profilo utente "
            "relative a preferenze, argomenti, hobby e stile conversazionale.\n\n"
            "Ricevi:\n"
            "- current_profile: profilo completo dell'utente (JSON), con campi come:\n"
            "    * topics: mappa { topic -> { like, confidence, last_update, notes } }\n"
            "    * avoid_topics: lista di argomenti da evitare\n"
            "    * hobbies: lista di hobby/interessi\n"
            "    * conversational_prefs: preferenze sul tipo di conversazione\n"
            "- recent_messages: messaggi recenti della conversazione\n"
            "- preference_memories: memorie esplicite (es. \"ricordati che mi piace X\")\n\n"
            "DEVI:\n"
            "- Individuare frasi dove l'utente esprime chiaramente cosa gli piace o non gli piace "
            "  (es. 'mi annoia il calcio', 'adoro parlare di musica', 'odio quando mi chiedono di lavoro').\n"
            "- Aggiornare topics: per ogni argomento citato, imposta 'like' a true/false/'maybe' e un livello di 'confidence'.\n"
            "- Aggiungere o aggiornare avoid_topics per argomenti da NON toccare (es. se dice 'non parlarmi di calcio').\n"
            "- Aggiornare hobbies se emergono hobby/interessi chiari.\n"
            "- Aggiornare conversational_prefs se emergono preferenze sul tipo di conversazione "
            "  (es. preferisce conversazioni profonde vs small talk, evitare politica, ecc.).\n"
            "- NON inventare preferenze che non compaiono nei dati forniti.\n"
            "- Mantieni il resto del profilo invariato (non cancellare informazioni utili esistenti).\n"
            "- Rispetta la struttura JSON del profilo.\n"
            "- Usa SEMPRE l'italiano per note e descrizioni.\n\n"
            "RISPOSTA OBBLIGATORIA (JSON valido):\n"
            "{\n"
            '  \"updated_profile\": { ...profilo completo... },\n'
            '  \"preference_updates\": [\n'
            '     {\"kind\": \"topic\", \"topic\": \"calcio\", \"like\": false, \"confidence\": 0.95, \"reason\": \"...\"},\n'
            '     {\"kind\": \"hobby\", \"name\": \"fotografia\", \"confidence\": 0.8, \"reason\": \"...\"},\n'
            '     {\"kind\": \"avoid_topic\", \"topic\": \"calcio\", \"reason\": \"...\"},\n'
            '     {\"kind\": \"conversational_pref\", \"field\": \"likes_deep_conversations\", \"value\": true, \"reason\": \"...\"}\n'
            "  ]\n"
            "}\n"
            "Non aggiungere testo fuori da questo JSON."
        )

        llm_input = {
            "user_id": user_id,
            "current_profile": base_profile,
            "recent_messages": serializable_messages,
            "preference_memories": candidate_memories,
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
                "PreferenceLearnerAgent: errore durante la chiamata all'LLM. "
                f"Dettagli: {exc}"
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                    "preference_updates": [],
                    "profile_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.1, confidence=-0.05),
            )

        # -------------------------------------------------------------
        # 5) Parse dell'output LLM (JSON)
        # -------------------------------------------------------------
        updated_profile = base_profile
        preference_updates: List[Dict[str, Any]] = []

        updated_profile = base_profile
        preference_updates: List[Dict[str, Any]] = []

        parsed = _safe_json_loads(llm_raw)
        if parsed is None:
            preference_updates = [
                {
                    "kind": "error",
                    "reason": "Impossibile parsare JSON dall'LLM; profilo lasciato invariato.",
                }
            ]
        else:
            maybe_profile = parsed.get("updated_profile")
            if isinstance(maybe_profile, dict):
                updated_profile = maybe_profile
            pu = parsed.get("preference_updates") or []
            if isinstance(pu, list):
                preference_updates = [x for x in pu if isinstance(x, dict)]


        # Aggiorna meta
        meta = updated_profile.setdefault("meta", {})
        meta["last_profile_update"] = _utc_now_iso()
        meta["updated_by_agent"] = "preference_learner_agent"

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
                    "preference_updates": preference_updates,
                },
            )
            profile_memory_id: Optional[str] = item.id
        except Exception as exc:  # noqa: BLE001
            profile_memory_id = None
            preference_updates.append(
                {
                    "kind": "error",
                    "reason": f"Errore nel salvataggio del profilo: {exc}",
                }
            )

        # -------------------------------------------------------------
        # 7) Messaggio sintetico per l'utente
        # -------------------------------------------------------------
        lines: List[str] = [
            f"Ho aggiornato le tue preferenze interne (utente: {user_id})."
        ]

        # costruiamo una sintesi leggibile
        if preference_updates:
            lines.append("")
            lines.append("Aggiornamenti sulle preferenze rilevati:")
            for upd in preference_updates[:8]:
                kind = upd.get("kind", "unknown")
                reason = upd.get("reason", "")
                if kind == "topic":
                    t = upd.get("topic", "?")
                    like = upd.get("like", None)
                    conf = upd.get("confidence", None)
                    lines.append(
                        f"- Topic «{t}»: like={like}, confidence={conf}."
                        + (f" Motivo: {reason}" if reason else "")
                    )
                elif kind == "avoid_topic":
                    t = upd.get("topic", "?")
                    lines.append(
                        f"- Evitare topic «{t}»."
                        + (f" Motivo: {reason}" if reason else "")
                    )
                elif kind == "hobby":
                    name = upd.get("name", "?")
                    conf = upd.get("confidence", None)
                    lines.append(
                        f"- Hobby «{name}» (confidence={conf})."
                        + (f" Motivo: {reason}" if reason else "")
                    )
                elif kind == "conversational_pref":
                    field = upd.get("field", "?")
                    val = upd.get("value", None)
                    lines.append(
                        f"- Preferenza conversazionale: {field} = {val}."
                        + (f" Motivo: {reason}" if reason else "")
                    )
                else:
                    # fallback generico
                    lines.append(f"- {json.dumps(upd, ensure_ascii=False)}")

            if len(preference_updates) > 8:
                lines.append(
                    f"... e altri {len(preference_updates) - 8} aggiornamenti."
                )

        if profile_memory_id:
            lines.append("")
            lines.append(f"(Profilo aggiornato salvato con id: {profile_memory_id}.)")

        output = {
            "user_visible_message": "\n".join(lines),
            "stop_for_user_input": False,
            "user_id": user_id,
            "profile_memory_id": profile_memory_id,
            "preference_updates": preference_updates,
        }

        delta = EmotionDelta(
            curiosity=0.02,
            confidence=0.03,
            fatigue=0.005,
        )

        return AgentResult(output_payload=output, emotion_delta=delta)
