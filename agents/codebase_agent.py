# agents/codebase_agent.py
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.agents_base import Agent, AgentResult, ACTIVE_REGISTRY
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
    Utility usata anche in altri agent LLM-based:
    - prova json.loads diretto,
    - se fallisce, prova a estrarre il primo blocco {...}.
    """
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

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


class CodebaseAgent(Agent):
    """
    File/Codebase Agent

    Scopi principali:
      - indicizzare il codice del progetto (file, dimensione, estensione),
      - rispondere a domande tipo: "dove viene usato X?",
      - generare un piano di refactoring sfruttando LLM + contesto di codice.

    Modalità (input_payload["mode"]):

      1) "index":
         {
           "mode": "index",
           "max_files": 500        # opzionale, clampato
         }

      2) "search":
         {
           "mode": "search",
           "query": "nome_funzione_o_token",
           "max_hits": 40          # opzionale
         }

      3) "refactor_plan":
         {
           "mode": "refactor_plan",
           "goal": "descrizione in italiano di cosa vuoi cambiare",
           "query": "token o nome funzione/variabile",   # opzionale
           "max_hits": 30                                 # opzionale
         }

    Output (a seconda della modalità):

      - user_visible_message: testo leggibile per l'utente
      - index_summary: info sull'indice creato (per mode=index)
      - search_results: lista di match (per mode=search / refactor_plan)
      - refactor_plan: struttura JSON della proposta di refactoring (per mode=refactor_plan)
    """

    name = "codebase_agent"
    description = (
        "Indicizza i file di codice del progetto, permette ricerche tipo "
        "'dove viene usato X?' e costruisce piani di refactoring guidati da LLM."
    )

    # ------------------------------------------------------------------ #
    #  Helpers: filesystem / indexing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _project_root() -> Path:
        # assumiamo che il repo sia la cartella padre rispetto a agents/
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _iter_code_files(root: Path) -> Iterable[Path]:
        """
        Itera sui file di codice del progetto, saltando dir rumorose.
        """
        skip_dirs = {
            ".git",
            ".idea",
            ".vscode",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            "out",
            "r_agents/out",
        }
        allowed_ext = {
            ".py",
            ".R",
            ".r",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".sql",
            ".sh",
        }

        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            # filtra dir da saltare
            dirnames[:] = [
                d for d in dirnames
                if d not in skip_dirs and not d.startswith(".")
            ]

            for fname in filenames:
                path = Path(dirpath) / fname
                if path.suffix in allowed_ext:
                    yield path

    def _build_index(
        self,
        memory: MemoryEngine,
        max_files: int = 500,
    ) -> Dict[str, Any]:
        """
        Costruisce un indice leggero dei file di codice:
        - path relativo,
        - dimensione in byte,
        - estensione.
        Lo salva in memoria GLOBAL/PROCEDURAL con key='code_index'.
        """
        root = self._project_root()
        files_info: List[Dict[str, Any]] = []
        count = 0

        for path in self._iter_code_files(root):
            rel = path.relative_to(root)
            try:
                size = path.stat().st_size
            except OSError:
                size = None

            files_info.append(
                {
                    "path": str(rel),
                    "size": size,
                    "ext": path.suffix,
                }
            )
            count += 1
            if count >= max_files:
                break

        index_obj = {
            "root": str(root),
            "num_files": len(files_info),
            "files": files_info,
        }

        try:
            memory.store_item(
                scope=MemoryScope.GLOBAL,
                type_=MemoryType.PROCEDURAL,
                key="code_index",
                content=json.dumps(index_obj, ensure_ascii=False),
                metadata={
                    "agent": self.name,
                    "num_files": len(files_info),
                },
            )
        except Exception:
            # se il salvataggio fallisce, usiamo comunque l'indice in uscita
            pass

        return index_obj

    def _search_occurrences(
        self,
        query: str,
        max_hits: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Cerca in tutti i file di codice il testo 'query'.
        Ritorna una lista di match:
          { "file": "...", "line_no": 42, "line": "..." }
        """
        root = self._project_root()
        q = query
        hits: List[Dict[str, Any]] = []

        if not q:
            return hits

        for path in self._iter_code_files(root):
            rel = path.relative_to(root)
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, start=1):
                        if q in line:
                            hits.append(
                                {
                                    "file": str(rel),
                                    "line_no": line_no,
                                    "line": line.rstrip("\n"),
                                }
                            )
                            if len(hits) >= max_hits:
                                return hits
            except OSError:
                continue

        return hits

    # ------------------------------------------------------------------ #
    #  Core _run_impl
    # ------------------------------------------------------------------ #

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        mode = (input_payload.get("mode") or "search").lower()

        if mode == "index":
            return self._run_index(input_payload, memory)

        if mode == "search":
            return self._run_search(input_payload, memory)

        if mode == "refactor_plan":
            return self._run_refactor_plan(
                input_payload=input_payload,
                context=context,
                memory=memory,
                llm=llm,
                emotional_state=emotional_state,
            )

        # default: se arriva un mode sconosciuto
        msg = (
            "CodebaseAgent: modalità non riconosciuta. Usa 'index', 'search' "
            "oppure 'refactor_plan'."
        )
        return AgentResult(
            output_payload={
                "user_visible_message": msg,
                "index_summary": None,
                "search_results": [],
                "refactor_plan": None,
                "stop_for_user_input": False,
            },
            emotion_delta=EmotionDelta(confidence=-0.02),
        )

    # ------------------------------------------------------------------ #
    #  mode = "index"
    # ------------------------------------------------------------------ #

    def _run_index(
        self,
        input_payload: Dict[str, Any],
        memory: MemoryEngine,
    ) -> AgentResult:
        max_files = int(input_payload.get("max_files", 500))
        max_files = max(50, min(max_files, 2000))

        index_obj = self._build_index(memory=memory, max_files=max_files)

        msg = (
            f"Ho indicizzato {index_obj['num_files']} file di codice nel progetto. "
            "Ora posso rispondere meglio a domande tipo 'dove viene usato X?'."
        )

        return AgentResult(
            output_payload={
                "user_visible_message": msg,
                "index_summary": index_obj,
                "search_results": [],
                "refactor_plan": None,
                "stop_for_user_input": False,
            },
            emotion_delta=EmotionDelta(
                curiosity=0.02,
                confidence=0.03,
            ),
        )

    # ------------------------------------------------------------------ #
    #  mode = "search"
    # ------------------------------------------------------------------ #

    def _run_search(
        self,
        input_payload: Dict[str, Any],
        memory: MemoryEngine,  # noqa: ARG002
    ) -> AgentResult:
        query = (input_payload.get("query") or "").strip()
        max_hits = int(input_payload.get("max_hits", 40))
        max_hits = max(5, min(max_hits, 200))

        if not query:
            msg = "Dimmi cosa vuoi cercare (es. 'where is used X?') impostando input_payload['query']."
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "index_summary": None,
                    "search_results": [],
                    "refactor_plan": None,
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(confidence=-0.02),
            )

        hits = self._search_occurrences(query=query, max_hits=max_hits)

        if not hits:
            msg = f"Non ho trovato occorrenze di '{query}' nei file di codice indicizzati."
        else:
            lines = [f"Ho trovato {len(hits)} occorrenze di '{query}' (mostro al massimo {max_hits}):", ""]
            for h in hits:
                lines.append(
                    f"- {h['file']}:{h['line_no']}: {h['line']}"
                )
            msg = "\n".join(lines)

        return AgentResult(
            output_payload={
                "user_visible_message": msg,
                "index_summary": None,
                "search_results": hits,
                "refactor_plan": None,
                "stop_for_user_input": False,
            },
            emotion_delta=EmotionDelta(
                curiosity=0.03 if hits else -0.01,
                confidence=0.02 if hits else -0.01,
            ),
        )

    # ------------------------------------------------------------------ #
    #  mode = "refactor_plan"
    # ------------------------------------------------------------------ #

    def _run_refactor_plan(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        goal = (input_payload.get("goal") or "").strip()
        query = (input_payload.get("query") or "").strip()
        max_hits = int(input_payload.get("max_hits", 30))
        max_hits = max(5, min(max_hits, 100))

        if not goal:
            msg = (
                "Per generare un piano di refactoring ho bisogno di un obiettivo chiaro. "
                "Imposta input_payload['goal'], ad esempio: "
                "\"separare la logica del Router in più moduli\"."
            )
            return AgentResult(
                output_payload={
                    "user_visible_message": msg,
                    "index_summary": None,
                    "search_results": [],
                    "refactor_plan": None,
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(confidence=-0.03),
            )

        hits: List[Dict[str, Any]] = []
        if query:
            hits = self._search_occurrences(query=query, max_hits=max_hits)

        # Costruisco input per l'LLM
        root = self._project_root()
        user_last = context.messages[-1].content if context.messages else ""

        llm_input = {
            "project_root": str(root),
            "refactor_goal": goal,
            "symbol_query": query,
            "search_hits": hits,
            "last_user_message": user_last,
            "emotional_state": {
                "curiosity": emotional_state.curiosity,
                "confidence": emotional_state.confidence,
                "fatigue": emotional_state.fatigue,
                "frustration": emotional_state.frustration,
            },
        }

        system_prompt = (
            "Sei un assistente che aiuta a pianificare refactoring in una codebase Python/R.\n"
            "Ti fornisco:\n"
            "- un obiettivo di refactoring (refactor_goal),\n"
            "- un eventuale simbolo/token da cercare (symbol_query),\n"
            "- alcuni match di ricerca nei file (search_hits),\n"
            "- l'ultimo messaggio utente.\n\n"
            "Compito:\n"
            "1) Proponi un piano di refactoring a passi numerati.\n"
            "2) Per ogni passo specifica file coinvolti, rischio e nota breve.\n"
            "3) Rispetta questo schema JSON:\n"
            "{\n"
            '  \"plan_summary\": \"riassunto sintetico in italiano\",\n'
            '  \"steps\": [\n'
            "    {\n"
            '      \"id\": \"step1\",\n'
            '      \"description\": \"cosa fare\",\n'
            '      \"files\": [\"path/relativo1.py\", \"...\"] ,\n'
            '      \"risk\": \"basso|medio|alto\",\n'
            '      \"estimation\": \"stima sforzo (es. bassa/2-3 ore)\",\n'
            '      \"notes\": \"dettagli opzionali\"\n'
            "    }\n"
            "  ],\n"
            '  \"notes\": \"eventuali note extra\"\n'
            "}\n\n"
            "Rispondi SOLO con JSON valido, senza testo fuori dal JSON."
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content=json.dumps(llm_input, ensure_ascii=False),
            )
        ]

        try:
            raw = llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=900,
            )
            parsed = _safe_json_loads(raw) or {}
        except Exception:
            parsed = {}

        plan_summary = parsed.get("plan_summary") or (
            "Piano di refactoring generato automaticamente sulla base dell'obiettivo fornito."
        )
        steps = parsed.get("steps") or []
        notes = parsed.get("notes") or ""

        # messaggio per l'utente
        lines: List[str] = []
        lines.append("Ti propongo questo piano di refactoring:")
        lines.append("")
        if isinstance(steps, list) and steps:
            for step in steps:
                sid = step.get("id") or "step"
                desc = step.get("description") or ""
                files = step.get("files") or []
                risk = step.get("risk") or "n.d."

                base = f"- {sid}: {desc}"
                if files:
                    base += f" (file coinvolti: {', '.join(files)})"
                base += f" [rischio: {risk}]"
                lines.append(base)
        else:
            lines.append("(Non sono riuscito a costruire un elenco dettagliato di passi.)")

        if notes:
            lines.append("")
            lines.append(f"Note aggiuntive: {notes}")

        user_msg = "\n".join(lines)

        # Salvo il piano in memoria procedurale a livello progetto (se disponibile)
        project_id = getattr(context, "project_id", None) or getattr(context, "current_project_id", None)
        refactor_memory_id: Optional[int] = None
        try:
            scope = MemoryScope.PROJECT if project_id else MemoryScope.GLOBAL
            metadata: Dict[str, Any] = {
                "agent": self.name,
                "goal": goal,
                "symbol_query": query,
            }
            if project_id:
                metadata["project_id"] = project_id

            item = memory.store_item(
                scope=scope,
                type_=MemoryType.PROCEDURAL,
                key="refactor_plan",
                content=json.dumps(
                    {"plan_summary": plan_summary, "steps": steps, "notes": notes},
                    ensure_ascii=False,
                ),
                metadata=metadata,
            )
            refactor_memory_id = item.id
        except Exception:
            refactor_memory_id = None

        delta = EmotionDelta(
            curiosity=0.04,
            confidence=0.04,
        )

        return AgentResult(
            output_payload={
                "user_visible_message": user_msg,
                "index_summary": None,
                "search_results": hits,
                "refactor_plan": {
                    "plan_summary": plan_summary,
                    "steps": steps,
                    "notes": notes,
                    "memory_id": refactor_memory_id,
                },
                "stop_for_user_input": False,
            },
            emotion_delta=delta,
        )


# Registrazione automatica se il registry globale è disponibile
if ACTIVE_REGISTRY is not None:
    ACTIVE_REGISTRY.register(CodebaseAgent())
