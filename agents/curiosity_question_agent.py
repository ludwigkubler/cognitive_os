# agents/curiosity_question_agent.py
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

# Riutilizziamo l'helper del profilo da user_profile_agent per garantire coerenza.
try:
    from agents.user_profile_agent import _ensure_base_profile
except ImportError:
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
                "updated_by_agent": "curiosity_question_agent",
                "notes": "",
            },
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CuriosityQuestionAgent(Agent):
    """
    CuriosityQuestionAgent

    Scopo:
      - Fare 1–3 domande nuove, sensate, personalizzate, per conoscere meglio l'utente.
      - Usare il profilo utente, lo stato emotivo e la conversazione recente.
      - Rispettare le preferenze:
          * scegliere temi che piacciono (topics.like = true, buona confidence)
          * evitare avoid_topics e topics con like=false
          * tenere conto delle open_questions già presenti (riusarle o chiuderle)

    Input atteso (input_payload):
      {
        "user_id": "user-1",             # opzionale, default context.user_id
        "max_messages": 20,              # opzionale, finestrella recente
        "max_questions": 3,              # opzionale, massimo di domande da proporre
        "user_profile": { ... },         # opzionale, se già fornito dal Router
        "force": false                   # opzionale, se true prova comunque a generare almeno 1 domanda
      }

    Output:
      - user_visible_message: testo con le domande da porre all'utente
      - questions: lista di domande (strings)
      - updated_profile: profilo aggiornato (open_questions/ recent_themes)
      - profile_memory_id: id del MemoryItem del profilo aggiornato (se salvato)
    """

    name = "curiosity_question_agent"
    description = (
        "Genera 1–3 domande personali curiose e rispettose, basate sul profilo "
        "utente (preferenze, hobby, valori) e sulla conversazione recente."
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
        # 1) Identifica utente e parametri
        # -------------------------------------------------------------
        user_id = input_payload.get("user_id") or getattr(context, "user_id", None)
        if not user_id:
            msg = (
                "CuriosityQuestionAgent: non riesco a determinare lo user_id. "
                "Serve context.user_id o input_payload['user_id']."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                    "questions": [],
                    "profile_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.02, confidence=-0.02),
            )

        max_messages = int(input_payload.get("max_messages", 20))
        max_questions = int(input_payload.get("max_questions", 3))
        force = bool(input_payload.get("force", False))

        max_messages = max(5, min(max_messages, 100))
        max_questions = max(1, min(max_questions, 5))

        # -------------------------------------------------------------
        # 2) Carica profilo utente (da input_payload o da memoria)
        # -------------------------------------------------------------
        profile_key = f"user_profile:{user_id}"

        raw_profile_from_payload = None
        if "user_profile" in input_payload:
            try:
                raw_profile_from_payload = json.dumps(input_payload["user_profile"], ensure_ascii=False)
            except Exception:
                raw_profile_from_payload = None

        if raw_profile_from_payload is not None:
            raw_profile = raw_profile_from_payload
        else:
            raw_profile = memory.load_item_content(
                key=profile_key,
                scope=MemoryScope.USER,
                type_=MemoryType.SEMANTIC,
            )

        base_profile = _ensure_base_profile(user_id=user_id, raw_profile=raw_profile)

        # -------------------------------------------------------------
        # 3) Preparazione del contesto per l'LLM
        # -------------------------------------------------------------
        # a) Ultimi N messaggi (per evitare di ripetere la stessa domanda della stessa sessione)
        recent_msgs = context.messages[-max_messages:]
        serializable_messages: List[Dict[str, Any]] = []
        for m in recent_msgs:
            serializable_messages.append(
                {
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
            )

        # b) Stato emotivo del sistema (usato solo come segnale, non per euristiche hard-coded)
        emo = {
            "curiosity": emotional_state.curiosity,
            "confidence": emotional_state.confidence,
            "fatigue": emotional_state.fatigue,
            "frustration": emotional_state.frustration,
        }

        # -------------------------------------------------------------
        # 4) Prompt all'LLM: genera domande personalizzate
        # -------------------------------------------------------------
        system_prompt = (
            "Sei il CuriosityQuestionAgent di un sistema cognitivo multi-agent.\n"
            "Il tuo compito è generare 1–3 domande personali, curiose e rispettose, "
            "per conoscere meglio l'utente, basandoti sul suo profilo e sulla conversazione recente.\n\n"
            "Informazioni a disposizione:\n"
            "- user_profile: profilo utente strutturato con campi come topics, avoid_topics, hobbies, values, "
            "  conversational_prefs, recent_themes, open_questions, ecc.\n"
            "- recent_messages: ultimo segmento di conversazione.\n"
            "- emotional_state: curiosità, fatica, ecc. del sistema.\n\n"
            "REGOLE IMPORTANTI:\n"
            "- Usa topics: scegli argomenti che l'utente gradisce (like=true, buona confidence) "
            "  e che non siano in avoid_topics.\n"
            "- NON fare domande su argomenti esplicitamente evitati:\n"
            "    * se un topic ha like=false, NON chiedere su quel topic;\n"
            "    * se un topic è in avoid_topics, NON chiedere su quel topic.\n"
            "- Considera hobbies e values come ottime fonti di domande (es. creatività, progetti personali, passioni).\n"
            "- Considera conversational_prefs: se likes_deep_conversations=true, puoi porre domande "
            "  più profonde e riflessive; se false, mantieni le domande più leggere.\n"
            "- Usa recent_themes per evitare di ripetere subito la stessa identica domanda già fatta.\n"
            "- Usa open_questions (if any): puoi riprendere domande ancora 'pending' oppure generarne di nuove.\n"
            "- Modula il numero e l'intensità delle domande in base a emotional_state:\n"
            "    * curiosity alta e fatigue bassa → più probabile fare 2–3 domande;\n"
            "    * fatigue alta → al massimo 1 domanda, o anche nessuna se inappropriato.\n"
            "- Se force=true nel payload, cerca comunque di generare almeno 1 domanda, "
            "  purché non violi avoid_topics o preferenze chiare.\n"
            "- Le domande devono essere in lingua ITALIANA, in tono rispettoso e naturale.\n\n"
            "RISPOSTA OBBLIGATORIA (JSON valido):\n"
            "{\n"
            '  \"questions_to_ask\": [\"domanda 1\", \"domanda 2\", ...],\n'
            '  \"updated_profile\": { ...profilo_completo... },\n'
            '  \"notes\": \"spiegazione sintetica di che tipo di curiosità stai esplorando\"\n'
            "}\n"
            "Se non è appropriato porre domande (es. perché non c'è abbastanza contesto), "
            "puoi restituire questions_to_ask come lista vuota, ma mantieni comunque la struttura JSON."
        )

        llm_input = {
            "user_id": user_id,
            "user_profile": base_profile,
            "recent_messages": serializable_messages,
            "emotional_state": emo,
            "max_questions": max_questions,
            "force": force,
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
                max_tokens=768,
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                "CuriosityQuestionAgent: errore durante la chiamata all'LLM. "
                f"Dettagli: {exc}"
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                    "questions": [],
                    "profile_memory_id": None,
                },
                emotion_delta=EmotionDelta(frustration=0.08, confidence=-0.05),
            )

        # -------------------------------------------------------------
        # 5) Parse dell'output dell'LLM
        # -------------------------------------------------------------
        questions: List[str] = []
        updated_profile = base_profile
        notes: str = ""

        try:
            parsed = json.loads(llm_raw)
            if isinstance(parsed, dict):
                qs = parsed.get("questions_to_ask") or []
                if isinstance(qs, list):
                    questions = [str(q).strip() for q in qs if str(q).strip()]
                maybe_profile = parsed.get("updated_profile")
                if isinstance(maybe_profile, dict):
                    updated_profile = maybe_profile
                n = parsed.get("notes")
                if isinstance(n, str):
                    notes = n
        except Exception:
            notes = "Impossibile parsare JSON dall'LLM; uso il profilo di base e nessuna domanda."

        # Aggiorna meta / last_profile_update
        meta = updated_profile.setdefault("meta", {})
        meta["last_profile_update"] = _utc_now_iso()
        meta["updated_by_agent"] = "curiosity_question_agent"

        # -------------------------------------------------------------
        # 6) Salvataggio del profilo aggiornato (open_questions, ecc.)
        # -------------------------------------------------------------
        profile_memory_id: Optional[str] = None
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
                    "curiosity_notes": notes,
                    "questions": questions,
                },
            )
            profile_memory_id = item.id
        except Exception as exc:  # noqa: BLE001
            # Non blocchiamo l'interazione, ma annotiamo nelle note
            notes += f" (Errore nel salvataggio del profilo: {exc})"

        # -------------------------------------------------------------
        # 7) Messaggio per l'utente
        # -------------------------------------------------------------
        if not questions:
            user_text = (
                "Per ora non ho una domanda personale sensata da porti, "
                "ma sto continuando a imparare dalle nostre conversazioni."
            )
            stop_for_user_input = False
        else:
            # Costruiamo un testo naturale con le domande
            lines: List[str] = []
            lines.append("Ti va di rispondere a qualche domanda su di te?")
            lines.append("")
            for i, q in enumerate(questions, start=1):
                lines.append(f"{i}. {q}")
            user_text = "\n".join(lines)
            # qui vogliamo effettivamente che l'utente risponda: fermiamo la pipeline
            stop_for_user_input = True

        output = {
            "user_visible_message": user_text,
            "stop_for_user_input": stop_for_user_input,
            "user_id": user_id,
            "questions": questions,
            "profile_memory_id": profile_memory_id,
            "curiosity_notes": notes,
        }

        # Curiosity leggermente consumata, confidence un po' su
        delta = EmotionDelta(
            curiosity=-0.02 if questions else -0.005,
            confidence=0.02,
        )

        return AgentResult(output_payload=output, emotion_delta=delta)
