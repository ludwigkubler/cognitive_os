from __future__ import annotations

import json
from typing import Any, Dict, List

from core.agents_base import Agent, AgentResult
from core.llm_provider import LLMProvider
from core.memory import MemoryEngine
from core.models import ConversationContext, EmotionalState, EmotionDelta


class SelfKnowledgeAgent(Agent):
    """
    Legge il profilo utente interno (user_profile:<user_id>) e produce
    una risposta testuale su ciò che il sistema SA di quell'utente
    (solo da memoria interna, niente mondo esterno).
    """

    name = "self_knowledge_agent"
    description = (
        "Riassume ciò che il sistema sa sull'utente a partire dal profilo interno."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # non usato: qui lavoriamo solo con JSON interno
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # 1) Identifico user_id
        user_id = input_payload.get("user_id") or getattr(context, "user_id", None)
        if not user_id:
            msg = (
                "Per ora non so chi sei perché non ho un 'user_id' nel contesto. "
                "Posso memorizzare qualcosa se mi dici come chiamarti."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(curiosity=0.01),
            )

        # 2) Carico il JSON del profilo
        raw_profile = memory.load_user_profile_json(user_id)
        if raw_profile is None:
            msg = (
                f"Al momento non ho ancora un profilo strutturato per '{user_id}'. "
                "So solo ciò che emerge dai messaggi recenti. "
                "Se vuoi, puoi raccontarmi di te e user_profile_agent costruirà il profilo."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(curiosity=0.02),
            )

        try:
            profile = json.loads(raw_profile)
        except Exception:
            msg = (
                "Ho trovato un profilo interno, ma sembra danneggiato o non leggibile. "
                "Posso ricostruirlo se mi racconti qualcosa di te."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(frustration=0.02, confidence=-0.02),
            )

        # 3) Costruisco una risposta leggibile a mano dal JSON
        display_name = profile.get("display_name") or user_id
        topics = profile.get("topics", {}) or {}
        avoid_topics = profile.get("avoid_topics", []) or []
        hobbies = profile.get("hobbies", []) or []
        values = profile.get("values", []) or []
        conv_prefs = profile.get("conversational_prefs", {}) or {}
        relationship = profile.get("relationship_with_system", {}) or {}
        recent_themes = profile.get("recent_themes", []) or []

        lines: List[str] = []
        lines.append(f"Ecco cosa so di te finora, {display_name}:")

        # Topics
        if topics:
            liked = []
            disliked = []
            neutral = []
            for name, info in topics.items():
                like = info.get("like")
                conf = info.get("confidence", 0.0)
                if like is True:
                    liked.append((name, conf))
                elif like is False:
                    disliked.append((name, conf))
                else:
                    neutral.append((name, conf))

            if liked:
                liked_str = ", ".join(
                    f"{name} (conf. {conf:.2f})" for name, conf in liked
                )
                lines.append(f"- Ti piacciono in particolare: {liked_str}.")
            if disliked:
                disliked_str = ", ".join(
                    f"{name} (conf. {conf:.2f})" for name, conf in disliked
                )
                lines.append(f"- Preferisci evitare: {disliked_str}.")
            if neutral:
                neutral_str = ", ".join(name for name, _ in neutral)
                lines.append(
                    f"- Ho segnato alcuni topic come 'neutrali' o poco chiari: {neutral_str}."
                )

        # Avoid topics
        if avoid_topics:
            avoid_str = ", ".join(avoid_topics)
            lines.append(f"- Non dovrei parlare di: {avoid_str}.")

        # Hobbies
        if hobbies:
            hob_str = ", ".join(
                f"{h.get('name')} (conf. {h.get('confidence', 0.0):.2f})"
                for h in hobbies
                if h.get("name")
            )
            if hob_str:
                lines.append(f"- Hobbies / interessi personali: {hob_str}.")

        # Valori
        if values:
            val_str = ", ".join(values)
            lines.append(f"- Valori che percepisco importanti per te: {val_str}.")

        # Conversational preferences
        if conv_prefs:
            prefs_lines = []
            if conv_prefs.get("likes_deep_conversations"):
                prefs_lines.append("ti piacciono le conversazioni profonde")
            if conv_prefs.get("likes_current_events"):
                prefs_lines.append("ti interessano i temi di attualità")
            avoid_politics = conv_prefs.get("avoid_politics")
            if avoid_politics is True:
                prefs_lines.append("preferisci evitare discussioni politiche")
            elif avoid_politics == "maybe":
                prefs_lines.append("sei incerto sulle discussioni politiche")
            if prefs_lines:
                lines.append("- Preferenze conversazionali: " + "; ".join(prefs_lines) + ".")

        # Relationship with system
        if relationship:
            trust = relationship.get("trust_level")
            comfort = relationship.get("comfort_level")
            notes = relationship.get("notes")
            rel_parts = []
            if isinstance(trust, (int, float)):
                rel_parts.append(f"livello di fiducia percepito {trust:.2f}")
            if isinstance(comfort, (int, float)):
                rel_parts.append(f"comfort nel parlare {comfort:.2f}")
            if notes:
                rel_parts.append(notes)
            if rel_parts:
                lines.append("- Relazione con il sistema: " + "; ".join(rel_parts) + ".")

        # Recent themes
        if recent_themes:
            rt_str = ", ".join(
                t.get("topic") for t in recent_themes if t.get("topic")
            )
            if rt_str:
                lines.append(f"- Temi recenti di cui abbiamo parlato: {rt_str}.")

        # Se non ho raccolto nulla di sostanziale
        if len(lines) == 1:
            lines.append(
                "Per ora il profilo è quasi vuoto: so solo che stai parlando con me, "
                "ma non ho ancora molte informazioni strutturate su di te."
            )

        msg = "\n".join(lines)

        output_payload: Dict[str, Any] = {
            "user_visible_message": msg,
            "stop_for_user_input": False,
            "user_id": user_id,
            "user_profile_json": profile,
        }

        delta = EmotionDelta(
            curiosity=0.01,
            confidence=0.02,
        )
        return AgentResult(output_payload=output_payload, emotion_delta=delta)

 