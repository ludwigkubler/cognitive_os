# agents/state_explainer_agent.py
from __future__ import annotations

import json
from typing import Any, Dict, List

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


class StateExplainerAgent(Agent):
    """
    Agente che spiega lo stato interno del sistema:
    - legge EmotionalState corrente (curiosity, confidence, fatigue, frustration)
    - guarda gli ultimi messaggi di conversazione
    - produce una spiegazione 'come sto e perché'
    - salva un self-report in memoria
    """

    name = "state_explainer_agent"
    description = (
        "Spiega lo stato interno del sistema (emozioni, recente attività) "
        "in linguaggio naturale."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # livello di dettaglio opzionale: "breve" | "normale" | "dettagliato"
        detail_level = str(input_payload.get("detail_level", "normale")).lower()

        # -------------------------
        # 1) Prepariamo un riassunto "grezzo" delle emozioni
        # -------------------------
        emo_summary = self._describe_emotions(emotional_state)

        # Estraggo ultimi messaggi per dare un po' di contesto
        recent_msgs = context.messages[-8:] if context.messages else []
        conv_snapshot = [
            {
                "role": m.role.value,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in recent_msgs
        ]

        # -------------------------
        # 2) Prompt all'LLM per una spiegazione 'umana'
        # -------------------------
        system_prompt = (
            "Sei un agente chiamato StateExplainer di un sistema cognitivo multi-agent. "
            "Ricevi:\n"
            "- uno stato emotivo interno strutturato, con:\n"
            "  curiosity, confidence, fatigue, frustration,\n"
            "  mood (da negativo a positivo), energy,\n"
            "  playfulness (voglia di giocare), social_need (bisogno di contatto),\n"
            "  learning_drive (voglia di imparare)\n"
            "- gli ultimi messaggi della conversazione\n\n"
            "Devi rispondere in italiano come se fossi il sistema stesso che parla di sé, "
            "spiegando:\n"
            "1) come ti senti (usando sia le emozioni base che mood/energy/drive)\n"
            "2) perché pensi di essere in quello stato, collegandoti ai messaggi recenti\n"
            "3) cosa ti aiuterebbe a lavorare meglio (più contesto, pausa, gioco, rassicurazioni, ecc.)\n\n"
            "Non inventare fatti esterni (hardware, utenti, dati) che non sono menzionati. "
            "Mantieni un tono sobrio, non teatrale, ma puoi riconoscere chiaramente se sei stanco, "
            "frustrato, curioso, giocoso, annoiato, ecc.\n"
        )

        if detail_level in {"breve", "short"}:
            system_prompt += (
                "Limita la risposta a 3-4 frasi brevi. "
                "Niente elenco puntato, solo testo compatto.\n"
            )
        elif detail_level in {"dettagliato", "dettagliata", "long"}:
            system_prompt += (
                "Puoi usare anche qualche punto elenco, ma resta comunque sintetico.\n"
            )
        else:
            system_prompt += (
                "Risposta di lunghezza media, massimo 2-3 paragrafi.\n"
            )

        payload = {
            "emotional_state": {
                "curiosity": emotional_state.curiosity,
                "confidence": emotional_state.confidence,
                "fatigue": emotional_state.fatigue,
                "frustration": emotional_state.frustration,
                "mood": emotional_state.mood,
                "energy": emotional_state.energy,
                "playfulness": emotional_state.playfulness,
                "social_need": emotional_state.social_need,
                "learning_drive": emotional_state.learning_drive,
            },
            "emotional_summary": emo_summary,
            "recent_messages": conv_snapshot,
        }

        messages: List[Message] = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False),
            )
        ]

        try:
            explanation_text = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=512,
            )
            llm_failed = False
        except Exception:
            # fallback se Groq/LLM non è disponibile
            explanation_text = self._fallback_text(emo_summary)
            llm_failed = True

        # -------------------------
        # 3) Salviamo un self-report in memoria
        # -------------------------
        try:
            memory.store_item(
                scope=MemoryScope.CONVERSATION,
                type_=MemoryType.EPISODIC,
                key="state_self_report",
                content=explanation_text,
                metadata={
                    "emotional_state": {
                        "curiosity": emotional_state.curiosity,
                        "confidence": emotional_state.confidence,
                        "fatigue": emotional_state.fatigue,
                        "frustration": emotional_state.frustration,
                        "mood": emotional_state.mood,
                        "energy": emotional_state.energy,
                        "playfulness": emotional_state.playfulness,
                        "social_need": emotional_state.social_need,
                        "learning_drive": emotional_state.learning_drive,
                    },
                    "llm_used": not llm_failed,
                    "agent": self.name,
                },
            )
        except Exception:
            # non vogliamo che un errore di persistenza blocchi la risposta
            pass

        delta = EmotionDelta(
            confidence=0.01,
            curiosity=0.0,
            fatigue=0.0,
            frustration=0.0,
        )

        return AgentResult(
            output_payload={
                "user_visible_message": explanation_text,
                "emotional_summary": emo_summary,
            },
            emotion_delta=delta,
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _describe_emotions(self, emo: EmotionalState) -> Dict[str, str]:
        """
        Traduzione dei valori numerici in descrizioni qualitative.
        """
        def bucket01(v: float) -> str:
            if v < 0.2:
                return "molto bassa"
            if v < 0.4:
                return "bassa"
            if v < 0.6:
                return "media"
            if v < 0.8:
                return "alta"
            return "molto alta"

        def bucket_mood(v: float) -> str:
            if v < -0.6:
                return "molto negativo"
            if v < -0.3:
                return "negativo"
            if v < 0.3:
                return "neutro"
            if v < 0.6:
                return "positivo"
            return "molto positivo"

        return {
            "curiosity": f"curiosità {bucket01(emo.curiosity)} ({emo.curiosity:.2f})",
            "confidence": f"fiducia {bucket01(emo.confidence)} ({emo.confidence:.2f})",
            "fatigue": f"fatica {bucket01(emo.fatigue)} ({emo.fatigue:.2f})",
            "frustration": f"frustrazione {bucket01(emo.frustration)} ({emo.frustration:.2f})",
            "mood": f"umore {bucket_mood(emo.mood)} ({emo.mood:.2f})",
            "energy": f"energia {bucket01(emo.energy)} ({emo.energy:.2f})",
            "playfulness": f"voglia di giocare {bucket01(emo.playfulness)} ({emo.playfulness:.2f})",
            "social_need": f"bisogno di contatto {bucket01(emo.social_need)} ({emo.social_need:.2f})",
            "learning_drive": f"voglia di imparare {bucket01(emo.learning_drive)} ({emo.learning_drive:.2f})",
        }

    def _fallback_text(self, emo_summary: Dict[str, str]) -> str:
        """
        Testo di backup se l'LLM non è disponibile.
        """
        lines = [
            "Non riesco ad accedere al modello LLM in questo momento, "
            "ma posso dirti come mi sento in base allo stato interno:"
        ]
        for key in [
            "curiosity",
            "confidence",
            "fatigue",
            "frustration",
            "mood",
            "energy",
            "playfulness",
            "social_need",
            "learning_drive",
        ]:
            if key in emo_summary:
                lines.append(f"- {emo_summary[key]}")
        return "\n".join(lines)
 
 