# agents/architect_agent.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    new_id,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


AgentType = Literal["python", "r"]
LifecycleState = Literal["draft", "test", "active", "deprecated"]


@dataclass
class ToolConfig:
    """
    Descrizione logica di un tool che l'agent potrà usare.
    Per ora è solo metadata; in futuro puoi collegarlo a funzioni reali.
    """

    name: str
    description: str
    params_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """
    Config JSON che finisce dentro agent_definitions.config.

    Questo è il "cuore" della AgentDefinition evoluta:
    - type: "python" / "r"
    - module/class_name per Python
    - r_script_path per R
    - system_prompt_template / tools / default_parameters
    """

    type: AgentType = "python"

    # Python-specific
    module: Optional[str] = None
    class_name: Optional[str] = None

    # R-specific
    r_script_path: Optional[str] = None

    # Prompting / behavior
    system_prompt_template: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata opzionale
    tags: List[str] = field(default_factory=list)
    notes: str = ""


def _safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# 1) ARCHITECT
# ----------------------------------------------------------------------
class ArchitectAgent(Agent):
    """
    Genera una nuova AgentDefinition logica e la salva nella tabella
    agent_definitions tramite MemoryEngine.save_agent_definition().

    Input atteso (input_payload):
      - user_request: descrizione naturale del nuovo agent
      - preferred_type: "python" / "r" (opzionale, default "python")
      - suggested_name: nome suggerito (opzionale)
      - parent_id: id di un agent "genitore" (facoltativo, per genealogia)
    """

    name = "architect_agent"
    description = "Progetta nuove AgentDefinition (python / R) a partire da richieste utente."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        user_request = input_payload.get("user_request", "").strip()
        preferred_type: AgentType = input_payload.get("preferred_type", "python")  # type: ignore[assignment]
        if preferred_type not in ("python", "r"):
            preferred_type = "python"

        suggested_name = input_payload.get("suggested_name")
        parent_id = input_payload.get("parent_id")

        # ------------------------------------------------------------------
        # 1) Generiamo un id e un nome base robusto
        # ------------------------------------------------------------------
        agent_id = new_id()
        short_id = agent_id.split("-")[0]

        if suggested_name:
            base_name = suggested_name.strip().lower().replace(" ", "_")
        else:
            base_name = f"custom_agent_{short_id}"

        # ------------------------------------------------------------------
        # 2) Proviamo a farci aiutare dall'LLM per una config ricca
        # ------------------------------------------------------------------
        system_prompt = (
            "Sei un Architect per un sistema multi-agent. "
            "Devi definire un NUOVO agent a partire dalla richiesta utente.\n\n"
            "Rispondi SOLO con un JSON valido con lo schema minimo:\n"
            "{\n"
            '  "name": "nome_snake_case",\n'
            '  "description": "testo descrittivo",\n'
            '  "type": "python" o "r",\n'
            '  "module": "per_agent_python.esempio" (se type=python),\n'
            '  "class_name": "NomeClasseAgent" (se type=python),\n'
            '  "r_script_path": "r_agents/esempio.R" (se type=r),\n'
            '  "system_prompt_template": "istruzioni per l\'agent",\n'
            '  "tools": [\n'
            "    {\"name\": \"tool_name\", \"description\": \"...\", \"params_schema\": {\"param\": \"type\"}}\n"
            "  ],\n"
            '  "default_parameters": { ... },\n'
            '  "tags": ["tag1", "tag2"],\n'
            '  "notes": "note opzionali"\n'
            "}\n"
            "Non aggiungere testo fuori dal JSON."
        )

        from core.models import Message, MessageRole  # type: ignore

        llm_input = {
            "user_request": user_request,
            "preferred_type": preferred_type,
            "suggested_name": base_name,
        }

        llm_messages = [
            Message(role=MessageRole.USER, content=json.dumps(llm_input, ensure_ascii=False))
        ]

        raw = ""
        llm_config: Optional[Dict[str, Any]] = None
        try:
            raw = llm.generate(system_prompt=system_prompt, messages=llm_messages, max_tokens=1024)
            llm_config = _safe_json_loads(raw)
        except Exception:  # noqa: BLE001
            llm_config = None

        # ------------------------------------------------------------------
        # 3) Costruiamo l'AgentConfig a partire da LLM o fallback deterministico
        # ------------------------------------------------------------------
        if llm_config and isinstance(llm_config, dict):
            name_from_llm = llm_config.get("name") or base_name
            name = str(name_from_llm).strip().lower().replace(" ", "_")

            cfg = AgentConfig(
                type=llm_config.get("type", preferred_type),
                module=llm_config.get("module"),
                class_name=llm_config.get("class_name"),
                r_script_path=llm_config.get("r_script_path"),
                system_prompt_template=llm_config.get("system_prompt_template", ""),
                tools=llm_config.get("tools", []) or [],
                default_parameters=llm_config.get("default_parameters", {}) or {},
                tags=llm_config.get("tags", []) or [],
                notes=llm_config.get("notes", "") or "",
            )
            description = llm_config.get("description", user_request or f"Agent generato per: {base_name}")
        else:
            # Fallback completamente deterministico, senza dipendere dall'LLM
            name = base_name
            if preferred_type == "python":
                module = f"agents.{name}"
                class_name = "".join(part.capitalize() for part in name.split("_"))
                r_script_path = None
            else:
                module = None
                class_name = None
                r_script_path = f"r_agents/{name}.R"

            cfg = AgentConfig(
                type=preferred_type,
                module=module,
                class_name=class_name,
                r_script_path=r_script_path,
                system_prompt_template=(
                    "Sei un agent monofunzionale nel sistema cognitive_os. "
                    "Segui rigorosamente lo scopo per cui sei stato creato. "
                    "Non fare routing, non creare altri agent. "
                    "Rispondi nel modo più chiaro e strutturato possibile."
                ),
                tools=[],
                default_parameters={},
                tags=["auto_generated"],
                notes="Definizione generata senza LLM (fallback).",
            )
            description = user_request or f"Agent generato automaticamente ({preferred_type})."

        # ------------------------------------------------------------------
        # 4) Costruiamo la AgentDefinition logica (dict) + salvataggio
        # ------------------------------------------------------------------
        definition: Dict[str, Any] = {
            "id": agent_id,
            "name": name,
            "description": description,
            "config": asdict(cfg),
            "is_active": False,  # sempre bozza iniziale
            "parent_id": parent_id,
            "lifecycle_state": "draft",
            # created_at verrà impostato in save_agent_definition se mancante
        }

        # Persistiamo su SQLite tramite MemoryEngine
        try:
            memory.save_agent_definition(definition)
            saved_ok = True
        except Exception as exc:  # noqa: BLE001
            saved_ok = False
            definition["save_error"] = str(exc)

        # ------------------------------------------------------------------
        # 5) Output user-visible
        # ------------------------------------------------------------------
        msg_lines = [
            "Ho creato una nuova AgentDefinition in stato 'draft'.",
            f"- id: {agent_id}",
            f"- name: {name}",
            f"- type: {cfg.type}",
        ]
        if cfg.type == "python":
            msg_lines.append(f"- module: {cfg.module}")
            msg_lines.append(f"- class_name: {cfg.class_name}")
        else:
            msg_lines.append(f"- r_script_path: {cfg.r_script_path}")

        if not saved_ok:
            msg_lines.append("\n⚠️ Attenzione: si è verificato un errore nel salvataggio su memoria.")

        user_message = "\n".join(msg_lines)

        output = {
            "user_visible_message": user_message,
            "stop_for_user_input": False,
            "agent_definition": definition,
        }

        delta = EmotionDelta(curiosity=0.04, confidence=0.03)
        return AgentResult(output_payload=output, emotion_delta=delta)


# ----------------------------------------------------------------------
# 2) VALIDATOR
# ----------------------------------------------------------------------
class ValidatorAgent(Agent):
    """
    Valida l'ultima AgentDefinition (o una specificata) e, se richiesto,
    la promuove da draft → test.

    Input atteso:
      - target_id: opzionale, id specifico di AgentDefinition
      - min_description_len: opzionale, default 20
      - require_prompt: bool, default True
      - auto_promote_to_test: bool, default False
    """

    name = "validator_agent"
    description = "Valida coerenza e completezza di una AgentDefinition (draft/test)."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        if not hasattr(memory, "list_agent_definitions"):
            return AgentResult(
                output_payload={
                    "user_visible_message": "Validator: nessun supporto per agent_definitions in questa memoria.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        defs = memory.list_agent_definitions()
        if not defs:
            return AgentResult(
                output_payload={
                    "user_visible_message": "Validator: nessuna AgentDefinition trovata.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        target_id = input_payload.get("target_id")
        candidate: Optional[Dict[str, Any]] = None
        if target_id:
            for d in defs:
                if d["id"] == target_id:
                    candidate = d
                    break
        else:
            candidate = defs[-1]  # ultima definizione

        if candidate is None:
            return AgentResult(
                output_payload={
                    "user_visible_message": f"Validator: AgentDefinition con id '{target_id}' non trovata.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        # Parametri di validazione
        min_description_len = int(input_payload.get("min_description_len", 20))
        require_prompt = bool(input_payload.get("require_prompt", True))
        auto_promote = bool(input_payload.get("auto_promote_to_test", False))

        cfg_dict = candidate.get("config", {}) or {}
        cfg = AgentConfig(
            type=cfg_dict.get("type", "python"),
            module=cfg_dict.get("module"),
            class_name=cfg_dict.get("class_name"),
            r_script_path=cfg_dict.get("r_script_path"),
            system_prompt_template=cfg_dict.get("system_prompt_template", ""),
            tools=cfg_dict.get("tools", []) or [],
            default_parameters=cfg_dict.get("default_parameters", {}) or {},
            tags=cfg_dict.get("tags", []) or [],
            notes=cfg_dict.get("notes", "") or "",
        )

        issues: List[str] = []

        # 1) Tipo valido
        if cfg.type not in ("python", "r"):
            issues.append(f"type non valido: {cfg.type!r}")

        # 2) Binding per tipo
        if cfg.type == "python":
            if not cfg.module or not cfg.class_name:
                issues.append("Per type='python' servono module e class_name non vuoti.")
        else:
            if not cfg.r_script_path:
                issues.append("Per type='r' serve r_script_path non vuoto.")

        # 3) Descrizione minima
        desc = (candidate.get("description") or "").strip()
        if len(desc) < min_description_len:
            issues.append(
                f"Descrizione troppo corta (len={len(desc)}, minimo richiesto={min_description_len})."
            )

        # 4) Prompt di sistema minimo
        if require_prompt and len(cfg.system_prompt_template.strip()) < 10:
            issues.append("system_prompt_template troppo corto o mancante.")

        # 5) Nome
        name = candidate.get("name", "")
        if not name or " " in name:
            issues.append("name mancante o contiene spazi; usa snake_case.")

        valid = len(issues) == 0

        # Auto-promozione draft → test (solo se valido)
        previous_state = candidate.get("lifecycle_state", "draft")
        new_state = previous_state
        if valid and auto_promote and previous_state == "draft":
            new_state = "test"
            candidate["lifecycle_state"] = new_state
            candidate["is_active"] = False  # ancora in test
            candidate["config"] = asdict(cfg)
            try:
                memory.save_agent_definition(candidate)
            except Exception:
                issues.append("Errore durante save_agent_definition in auto-promozione.")

        # Output per l'utente
        lines = [
            f"Validator su AgentDefinition '{name}' (id={candidate['id']})",
            f"- type: {cfg.type}",
            f"- stato: {previous_state} → {new_state}",
            "",
        ]

        if valid:
            lines.append("✅ Validazione: OK (nessun problema bloccante).")
        else:
            lines.append("❌ Validazione: sono emersi problemi:")

        for issue in issues:
            lines.append(f"  - {issue}")

        output = {
            "user_visible_message": "\n".join(lines),
            "stop_for_user_input": False,
            "validation_ok": valid,
            "issues": issues,
            "candidate_id": candidate["id"],
            "new_lifecycle_state": new_state,
        }

        delta = EmotionDelta(confidence=0.03 if valid else -0.02)
        return AgentResult(output_payload=output, emotion_delta=delta)


# ----------------------------------------------------------------------
# 3) CRITIC
# ----------------------------------------------------------------------
class CriticAgent(Agent):
    """
    CriticAgent è l'ultimo step della pipeline:
    - legge una AgentDefinition (tipicamente già validata),
    - opzionalmente controlla alcune condizioni banali,
    - la può promuovere in 'active'.

    Input atteso:
      - target_id: opzionale, id da promuovere (default: ultima definizione)
      - promote_to_active: bool, default False
      - force_state: opzionale, se vuoi forzare uno stato specifico
    """

    name = "critic_agent"
    description = "Decide se promuovere una AgentDefinition a stato 'active'."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,  # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        if not hasattr(memory, "list_agent_definitions"):
            return AgentResult(
                output_payload={
                    "user_visible_message": "Critic: nessun supporto per agent_definitions in questa memoria.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        defs = memory.list_agent_definitions()
        if not defs:
            return AgentResult(
                output_payload={
                    "user_visible_message": "Critic: nessuna AgentDefinition trovata.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        target_id = input_payload.get("target_id")
        promote_to_active = bool(input_payload.get("promote_to_active", False))
        force_state = input_payload.get("force_state")

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
                    "user_visible_message": f"Critic: AgentDefinition con id '{target_id}' non trovata.",
                    "stop_for_user_input": False,
                },
                emotion_delta=EmotionDelta(),
            )

        name = candidate.get("name", "<unknown>")
        prev_state = candidate.get("lifecycle_state", "draft")
        prev_active = bool(candidate.get("is_active", False))

        # logica minimale: promuovi a active solo se è almeno in test
        new_state = prev_state
        new_active = prev_active

        if force_state in ("draft", "test", "active", "deprecated"):
            new_state = force_state  # override
            new_active = force_state == "active"
        elif promote_to_active:
            if prev_state not in ("test", "active"):
                # non promuovere da draft direttamente a active (per ora)
                new_state = "test"
                new_active = False
            else:
                new_state = "active"
                new_active = True

        candidate["lifecycle_state"] = new_state
        candidate["is_active"] = new_active

        # Scriviamo su memoria
        try:
            memory.save_agent_definition(candidate)
            save_ok = True
        except Exception as exc:  # noqa: BLE001
            save_ok = False
            save_error = str(exc)
        else:
            save_error = ""

        lines = [
            f"Critic su AgentDefinition '{name}' (id={candidate['id']})",
            f"- stato: {prev_state} → {new_state}",
            f"- is_active: {prev_active} → {new_active}",
        ]

        if promote_to_active or force_state:
            if save_ok:
                lines.append("✅ Stato aggiornato e salvato in memoria.")
            else:
                lines.append(f"⚠️ Errore nel salvataggio: {save_error}")
        else:
            lines.append("ℹ️ Nessuna promozione richiesta (solo valutazione).")

        output = {
            "user_visible_message": "\n".join(lines),
            "stop_for_user_input": False,
            "candidate_id": candidate["id"],
            "old_state": prev_state,
            "new_state": new_state,
            "is_active": new_active,
            "save_ok": save_ok,
        }

        delta = EmotionDelta(
            confidence=0.04 if save_ok else -0.02,
            curiosity=0.01,
        )
        return AgentResult(output_payload=output, emotion_delta=delta)
