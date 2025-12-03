# agents/codegen_agent.py
from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import EmotionalState, EmotionDelta, ConversationContext
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class CodegenAgent(Agent):
    """
    Genera codice sorgente per nuovi agent a partire da una AgentDefinition
    salvata in agent_definitions (MemoryEngine / SQLite).

    Supporta:
      - type="python"  → file .py in agents/
      - type="r"       → file .R in r_agents/

    Non registra l'agent nel registry (ci pensa il loader dinamico),
    ma aggiorna la AgentDefinition con informazioni sul file generato.
    """

    name = "codegen_agent"
    description = "Genera il codice sorgente (.py / .R) per nuove AgentDefinition."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        # ------------------------------------------------------------------
        # 1) Recupera le AgentDefinition
        # ------------------------------------------------------------------
        if not hasattr(memory, "list_agent_definitions"):
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "CodegenAgent: MemoryEngine non espone list_agent_definitions()."
                    ),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        defs = memory.list_agent_definitions()
        if not defs:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        "CodegenAgent: nessuna AgentDefinition trovata in memoria."
                    ),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        # target_id opzionale, altrimenti prendo l'ultima definizione
        target_id = input_payload.get("target_id")
        candidate: Optional[Dict[str, Any]] = None
        if target_id:
            for d in defs:
                if d["id"] == target_id:
                    candidate = d
                    break
        else:
            candidate = defs[-1]

        if candidate is None:
            return AgentResult(
                output_payload={
                    "user_visible_message": (
                        f"CodegenAgent: AgentDefinition con id '{target_id}' non trovata."
                    ),
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        cfg: Dict[str, Any] = candidate.get("config", {}) or {}

        agent_type = cfg.get("type", "python")
        if agent_type not in ("python", "r"):
            agent_type = "python"

        base_dir_python = input_payload.get("base_dir_python", "agents")
        base_dir_r = input_payload.get("base_dir_r", "r_agents")
        overwrite = bool(input_payload.get("overwrite_existing", False))
        dry_run = bool(input_payload.get("dry_run", False))

        # ------------------------------------------------------------------
        # 2) Genera codice in base al tipo
        # ------------------------------------------------------------------
        if agent_type == "python":
            result = self._generate_python_agent(
                definition=candidate,
                cfg=cfg,
                base_dir=base_dir_python,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        else:
            result = self._generate_r_agent(
                definition=candidate,
                cfg=cfg,
                base_dir=base_dir_r,
                overwrite=overwrite,
                dry_run=dry_run,
            )

        # Se abbiamo aggiornato la definizione (es. module/class_name/r_script_path),
        # riscriviamola su SQLite
        if result["updated_definition"]:
            try:
                memory.save_agent_definition(candidate)
            except Exception as exc:  # noqa: BLE001
                result["messages"].append(
                    f"⚠️ Errore nel salvataggio aggiornato della AgentDefinition: {exc}"
                )

        # ------------------------------------------------------------------
        # 3) Costruisci messaggio per l'utente
        # ------------------------------------------------------------------
        lines: List[str] = []
        lines.append(f"Codegen su AgentDefinition '{candidate['name']}' (id={candidate['id']})")
        lines.extend(result["messages"])

        user_msg = "\n".join(lines)

        output = {
            "user_visible_message": user_msg,
            "stop_for_user_input": False,
            "file_path": result.get("file_path"),
            "agent_type": agent_type,
            "dry_run": dry_run,
        }

        delta = EmotionDelta(
            confidence=0.04 if result.get("file_created") else 0.0,
            curiosity=0.02,
        )
        return AgentResult(output_payload=output, emotion_delta=delta)

    # ------------------------------------------------------------------
    #  PYTHON
    # ------------------------------------------------------------------
    def _generate_python_agent(
        self,
        definition: Dict[str, Any],
        cfg: Dict[str, Any],
        base_dir: str,
        overwrite: bool,
        dry_run: bool,
    ) -> Dict[str, Any]:
        messages: List[str] = []
        updated_definition = False

        name = definition.get("name", "custom_agent").strip().lower().replace(" ", "_")
        desc = definition.get("description", "").strip() or f"Agent generato automaticamente ({name})."

        # module / class_name: se mancanti, li deriviamo
        module = cfg.get("module")
        class_name = cfg.get("class_name")

        if not module:
            module = f"agents.{name}"
            cfg["module"] = module
            updated_definition = True

        if not class_name:
            class_name = "".join(part.capitalize() for part in name.split("_"))
            if not class_name.endswith("Agent"):
                class_name += "Agent"
            cfg["class_name"] = class_name
            updated_definition = True

        # path del file a partire dal module
        # es: agents.pdf_reader_agent → agents/pdf_reader_agent.py
        if module.startswith("agents."):
            rel = module[len("agents.") :]
        else:
            # modulo fuori dal package agents → lo trattiamo comunque sotto base_dir
            rel = module

        rel_path = rel.replace(".", os.sep) + ".py"
        file_path = os.path.join(base_dir, rel_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if os.path.exists(file_path) and not overwrite and not dry_run:
            messages.append(
                f"⚠️ Il file '{file_path}' esiste già. Usa overwrite_existing=True per sovrascrivere."
            )
            return {
                "file_created": False,
                "file_path": file_path,
                "messages": messages,
                "updated_definition": updated_definition,
            }

        system_prompt_template = cfg.get("system_prompt_template", "").strip()

        # corpo dell'agent Python generato
        code = self._render_python_agent_code(
            class_name=class_name,
            agent_name=name,
            description=desc,
            system_prompt_template=system_prompt_template,
        )

        if dry_run:
            messages.append("Modalità dry_run: non ho scritto nessun file.")
            messages.append("Anteprima del path di file:")
            messages.append(f"  - {file_path}")
        else:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                messages.append(f"✅ File Python generato: {file_path}")
            except Exception as exc:  # noqa: BLE001
                messages.append(f"❌ Errore durante la scrittura del file Python: {exc}")
                return {
                    "file_created": False,
                    "file_path": file_path,
                    "messages": messages,
                    "updated_definition": updated_definition,
                }

        return {
            "file_created": not dry_run,
            "file_path": file_path,
            "messages": messages,
            "updated_definition": updated_definition,
        }

    def _render_python_agent_code(
        self,
        class_name: str,
        agent_name: str,
        description: str,
        system_prompt_template: str,
    ) -> str:
        # normalizziamo il prompt in una stringa tripla
        prompt_escaped = system_prompt_template.replace('"""', '\\"""')

        # Template Python: già "usabile".
        # - legge user_message da input_payload o dal contesto,
        # - se SYSTEM_PROMPT è definito, chiama l'LLM in single-shot,
        # - altrimenti risponde in modo neutro.
        code = f'''# Auto-generated by CodegenAgent
from __future__ import annotations

from typing import Any, Dict

from core.agents_base import Agent, AgentResult
from core.models import EmotionalState, EmotionDelta, ConversationContext
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class {class_name}(Agent):
    """
    {description}
    (Generato automaticamente da CodegenAgent.)
    """

    name = "{agent_name}"
    description = "{description}"

    SYSTEM_PROMPT = """{prompt_escaped}"""

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # Recupera il messaggio utente principale:
        # 1) da input_payload["user_message"]
        # 2) altrimenti dall'ultimo messaggio USER nel contesto, se disponibile.
        user_message = input_payload.get("user_message") or (
            context.messages[-1].content if context.messages else ""
        )

        if self.SYSTEM_PROMPT:
            from core.models import Message, MessageRole  # import locale per evitare cicli

            messages = [
                Message(role=MessageRole.USER, content=user_message),
            ]
            llm_output = llm.generate(
                system_prompt=self.SYSTEM_PROMPT,
                messages=messages,
                max_tokens=512,
            )
            text = llm_output
        else:
            text = (
                "Sono un agent generato automaticamente. "
                "Non ho ancora una logica specifica oltre a questo messaggio di placeholder."
            )

        output = {{
            "user_visible_message": text,
            "stop_for_user_input": False,
        }}
        delta = EmotionDelta()
        return AgentResult(output_payload=output, emotion_delta=delta)
'''
        return textwrap.dedent(code)

    # ------------------------------------------------------------------
    #  R
    # ------------------------------------------------------------------
    def _generate_r_agent(
        self,
        definition: Dict[str, Any],
        cfg: Dict[str, Any],
        base_dir: str,
        overwrite: bool,
        dry_run: bool,
    ) -> Dict[str, Any]:
        messages: List[str] = []
        updated_definition = False

        name = definition.get("name", "custom_agent").strip().lower().replace(" ", "_")
        desc = definition.get("description", "").strip() or f"R agent generato automaticamente ({name})."

        r_script_path = cfg.get("r_script_path")
        if not r_script_path:
            r_script_path = f"{base_dir}/{name}.R"
            cfg["r_script_path"] = r_script_path
            updated_definition = True

        file_path = r_script_path
        # se è un path relativo, prepend base_dir se non già presente
        if not os.path.isabs(file_path) and not file_path.startswith(base_dir):
            file_path = os.path.join(base_dir, os.path.basename(file_path))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if os.path.exists(file_path) and not overwrite and not dry_run:
            messages.append(
                f"⚠️ Il file R '{file_path}' esiste già. Usa overwrite_existing=True per sovrascrivere."
            )
            return {
                "file_created": False,
                "file_path": file_path,
                "messages": messages,
                "updated_definition": updated_definition,
            }

        system_prompt_template = cfg.get("system_prompt_template", "").strip()

        code = self._render_r_agent_code(
            agent_name=name,
            description=desc,
            system_prompt_template=system_prompt_template,
        )

        if dry_run:
            messages.append("Modalità dry_run: non ho scritto nessun file R.")
            messages.append("Anteprima del path di file:")
            messages.append(f"  - {file_path}")
        else:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                messages.append(f"✅ File R generato: {file_path}")
            except Exception as exc:  # noqa: BLE001
                messages.append(f"❌ Errore durante la scrittura del file R: {exc}")
                return {
                    "file_created": False,
                    "file_path": file_path,
                    "messages": messages,
                    "updated_definition": updated_definition,
                }

        return {
            "file_created": not dry_run,
            "file_path": file_path,
            "messages": messages,
            "updated_definition": updated_definition,
        }

    def _render_r_agent_code(
        self,
        agent_name: str,
        description: str,
        system_prompt_template: str,
    ) -> str:
        # Template R con protocollo standard:
        # - legge JSON da stdin (o args),
        # - parse con jsonlite::fromJSON,
        # - restituisce un JSON "echo" arricchito.
        prompt_comment = system_prompt_template.replace("\n", " ")

        code = f"""# Auto-generated R agent script for '{agent_name}'
# Descrizione: {description}
# NOTE:
# - Legge JSON da stdin (o, se vuoto, da un argomento di fallback).
# - Restituisce JSON su stdout con un payload generico.

suppressPackageStartupMessages(library(jsonlite))

system_prompt <- "{prompt_comment}"

read_input_json <- function() {{
  # Prova a leggere tutto lo stdin
  input_lines <- tryCatch(
    readLines(con = "stdin", warn = FALSE),
    error = function(e) character(0)
  )
  txt <- paste(input_lines, collapse = "\\n")

  if (nzchar(txt)) {{
    return(txt)
  }}

  # Fallback: se non c'è niente su stdin, usa il primo argomento (se sembra un JSON)
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 0) {{
    return(args[[1]])
  }}

  return("")
}}

main <- function() {{
  raw_json <- read_input_json()

  if (!nzchar(raw_json)) {{
    result <- list(
      agent = "{agent_name}",
      status = "error",
      message = "Nessun input JSON ricevuto (stdin vuoto e nessun argomento).",
      input = NULL,
      system_prompt = system_prompt
    )
    cat(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null"))
    return(invisible(NULL))
  }}

  input_obj <- NULL
  parse_error <- NULL
  try {{
    input_obj <- jsonlite::fromJSON(raw_json, simplifyVector = TRUE)
  }} catch (e) {{
    parse_error <- as.character(e)
  }}

  if (!is.null(parse_error)) {{
    result <- list(
      agent = "{agent_name}",
      status = "error",
      message = "Errore nel parse del JSON di input.",
      parse_error = parse_error,
      raw = raw_json,
      system_prompt = system_prompt
    )
    cat(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null"))
    return(invisible(NULL))
  }}

  # Qui potrai implementare la logica analitica reale.
  # Per ora facciamo un semplice "echo" dell'input.
  result <- list(
    agent = "{agent_name}",
    status = "ok",
    message = "Skeleton R agent generated. Implementa qui la logica reale.",
    input = input_obj,
    system_prompt = system_prompt
  )

  cat(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null"))
}}

if (identical(environment(), globalenv())) {{
  main()
}}
"""
        return code
