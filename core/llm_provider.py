from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import os

from .models import Message, MessageRole


class LLMProvider(ABC):
    """
    Interfaccia astratta per i modelli LLM.
    """

    @abstractmethod
    def generate(self, system_prompt: str, messages: List[Message], **kwargs) -> str:
        """
        Ritorna il testo generato dall'LLM come stringa.
        `messages` è una lista di Message (role, content).
        """
        raise NotImplementedError


class SimpleEchoLLM(LLMProvider):
    """
    LLM finto per debug: ripete l'ultimo messaggio utente con un prefisso.
    Utile quando vuoi testare la pipeline senza consumare API.
    """

    def generate(self, system_prompt: str, messages: List[Message], **kwargs) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.role == MessageRole.USER:
                last_user = m.content
                break
        return f"[ECHO-DEMO] Mi hai detto: {last_user}"


class OpenAILLM(LLMProvider):
    """
    Implementazione LLMProvider per le API OpenAI (GPT-4.1 ecc.).
    Lasciata qui per compatibilità, ma nel tuo caso useremo GroqLLM.
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: Optional[str] = None,
    ) -> None:
        from openai import OpenAI  # import locale per evitare dipendenza se non usato

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAILLM: manca OPENAI_API_KEY nell'ambiente o api_key nel costruttore."
            )

        self.client = OpenAI(api_key="immettere l'API KEY di OpenAI")
        self.model = model

    def generate(self, system_prompt: str, messages: List[Message], **kwargs) -> str:
        api_messages = []

        # System prompt
        if system_prompt:
            api_messages.append(
                {"role": "system", "content": system_prompt},
            )

        # Messaggi di contesto
        for m in messages:
            if m.role in {MessageRole.ASSISTANT, MessageRole.AGENT}:
                role = "assistant"
            else:
                role = m.role.value
            api_messages.append({"role": role, "content": m.content})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            **kwargs,
        )
        return resp.choices[0].message.content


class GroqLLM(LLMProvider):
    """
    Implementazione LLMProvider per GroqCloud.
    Usa la libreria `groq` e il modello di default `llama-3.3-70b-versatile`.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ) -> None:
        from groq import Groq  # import locale per evitare dipendenza se non usato

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GroqLLM: manca GROQ_API_KEY nell'ambiente o api_key nel costruttore."
            )

        self.client = Groq(api_key="immettere l'API KEY di Groq")
        self.model = model

    def generate(self, system_prompt: str, messages: List[Message], **kwargs) -> str:
        api_messages = []

        # System prompt
        if system_prompt:
            api_messages.append(
                {"role": "system", "content": system_prompt},
            )

        # Messaggi di contesto
        for m in messages:
            if m.role in {MessageRole.ASSISTANT, MessageRole.AGENT}:
                role = "assistant"
            else:
                role = m.role.value
            api_messages.append({"role": role, "content": m.content})

        # vedi doc Groq: client.chat.completions.create(...)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            **kwargs,
        )
        return resp.choices[0].message.content
