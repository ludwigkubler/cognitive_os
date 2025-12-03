from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MessageRole,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class ChatAgent(Agent):
    """
    Agente conversazionale principale:
    - usa l'LLM provider per generare la risposta
    - costruisce il contesto usando la history della conversazione
    - include sempre il profilo utente (se esiste) nel prompt
    - chiama uno script R per salvare l'intera conversazione
      in .RData e .db (RSQLite).
    """

    name = "chat_agent"
    description = (
        "Agente generico per dialogare con l'utente, usando il profilo interno "
        "e loggando la conversazione in R."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        user_message: str = input_payload.get("user_message", "")

        # Recupero user_id dal contesto (convenzionalmente 'user-1' nel CLI)
        user_id = getattr(context, "user_id", "unknown")

        # Carico il profilo utente JSON, se esiste (stringa JSON)
        raw_profile = memory.load_user_profile_json(user_id)
        profile_for_prompt = None
        if raw_profile is not None:
            try:
                profile_for_prompt = json.loads(raw_profile)
            except Exception:
                profile_for_prompt = None

        # -----------------------------
        # 1) Costruisco il prompt per l'LLM
        # -----------------------------
        system_prompt_parts: List[str] = []

        system_prompt_parts.append(
            "Sei un assistente conversazionale che parla con un utente umano. "
            "Rispondi in modo chiaro, sintetico ma utile. "
            "Se l'utente non specifica altro, rispondi in italiano."
        )

        # Stato emotivo (facoltativo, ma utile per futuri adattamenti)
        system_prompt_parts.append(
            f"Stato interno approssimato: curiosità={emotional_state.curiosity:.2f}, "
            f"fiducia={emotional_state.confidence:.2f}, "
            f"fatica={emotional_state.fatigue:.2f}, "
            f"frustrazione={emotional_state.frustration:.2f}."
        )

        # Profilo utente (se disponibile)
        if profile_for_prompt is not None:
            # Importante: vincolo esplicito per evitare invenzioni sul mondo esterno
            system_prompt_parts.append(
                "Hai accesso a un profilo utente interno in formato JSON, che contiene "
                "informazioni esclusivamente su questa persona (preferenze, hobby, "
                "topic da evitare, ecc.). "
                "Quando l'utente chiede 'cosa sai di me', 'cosa sai di <nome>' o fa "
                "riferimento a sé stesso (es. 'Ludovico Kubler, padre di Sophie'), "
                "rispondi SOLO usando questo profilo interno e la conversazione "
                "corrente. NON usare conoscenza esterna sul mondo anche se esistono "
                "altre persone famose con lo stesso nome."
            )
            system_prompt_parts.append(
                "Ecco il profilo utente interno (JSON):\n"
                + json.dumps(profile_for_prompt, ensure_ascii=False)
            )
        else:
            system_prompt_parts.append(
                "Al momento non hai ancora un profilo utente strutturato; "
                "usa solo ciò che emerge dalla conversazione corrente."
            )

        system_prompt = "\n\n".join(system_prompt_parts)

        # Prendiamo una finestra degli ultimi N messaggi per il contesto
        max_history = 12
        history = context.messages[-max_history:]

        # L'LLM provider si aspetta una lista di 'Message' del nostro modello,
        # possiamo usare direttamente la history più l'ultimo user_message.
        from core.models import Message

        messages_for_llm: List[Message] = []
        for msg in history:
            messages_for_llm.append(msg)

        # L'orchestrator ha già aggiunto il messaggio utente al context
        # prima di chiamare questo agent, quindi lo troviamo in history.

        # Chiamata al modello
        llm_raw = llm.generate(
            system_prompt=system_prompt,
            messages=messages_for_llm,
            max_tokens=512,
        )

        reply_text = llm_raw.strip()

        # -----------------------------
        # 2) Preparo un job JSON per R con l'intera conversazione
        #    + la risposta che sto per dare
        # -----------------------------
        # Costruiamo un vettore di messaggi role/content/timestamp
        conv_msgs: List[Dict[str, Any]] = []
        for msg in context.messages:
            conv_msgs.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
            )

        # Aggiungiamo la risposta che sto generando ora,
        # così in R hai già user + assistant allineati.
        conv_msgs.append(
            {
                "role": MessageRole.ASSISTANT.value,
                "content": reply_text,
                "timestamp": None,  # l'orchestrator la metterà dopo, qui è solo log logico
            }
        )

        conversation_id = context.id

        job = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": conv_msgs,
        }

        # -----------------------------
        # 3) Chiamo Rscript conversation_logger.R
        # -----------------------------
        project_root = Path(__file__).resolve().parents[1]
        r_dir = project_root / "r_agents"
        script = r_dir / "conversation_logger.R"

        # Assicuriamoci che la cartella per gli output esista (es. r_agents/out/)
        out_dir = r_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        job["db_path"] = str(out_dir / "conversations.db")
        job["rdata_dir"] = str(out_dir)

        try:
            cmd = ["Rscript", str(script), json.dumps(job)]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                # Non bloccare la conversazione se il logging fallisce
                print("[WARN] conversation_logger.R error:", proc.stderr.strip())
        except FileNotFoundError:
            print("[WARN] Rscript non trovato nel PATH. Salvataggio in R disabilitato.")

        # -----------------------------
        # 4) Costruisco il risultato per l'orchestrator
        # -----------------------------
        output_payload = {
            "user_visible_message": reply_text,
            "stop_for_user_input": False,
            # per eventuali agent a valle che vogliono sapere il profilo
            "user_id": user_id,
            "user_profile_json": profile_for_prompt,
        }

        delta = EmotionDelta(
            confidence=0.03,
            curiosity=0.01,
        )
        return AgentResult(output_payload=output_payload, emotion_delta=delta)


