from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime


class MessageRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    AGENT = "agent"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Task:
    id: str
    description: str
    agent_name: str
    input_payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # --- nuovo: modello esplicito del piano ---
    # id di altri task da cui questo dipende (DAG logico del piano)
    depends_on: List[str] = field(default_factory=list)

    # politica di retry
    max_retries: int = 0          # quante volte posso ritentare dopo un errore
    retry_count: int = 0          # quante volte ho già ritentato

    # cost/budget (opzionali, solo meta)
    cost_estimate: Optional[float] = None   # costo stimato del task (token, tempo, €…)
    budget: Optional[float] = None          # budget massimo assegnato al task

    # tag generici, comodi per ExplanationAgent / UI
    tags: List[str] = field(default_factory=list)

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING
        self.updated_at = datetime.utcnow()

    def mark_done(self, result: Dict[str, Any]) -> None:
        self.status = TaskStatus.DONE
        self.result = result
        self.updated_at = datetime.utcnow()

    def mark_error(self, error: str) -> None:
        self.status = TaskStatus.ERROR
        self.error = error
        self.updated_at = datetime.utcnow()

@dataclass
class Plan:
    id: str
    tasks: List[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    current_index: int = 0

    # --- nuovo: meta generico del piano ---
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def get_next_task(self) -> Optional[Task]:
        """
        Restituisce il prossimo Task PENDING le cui dipendenze (depends_on)
        sono tutte completate (status DONE). Se nessun task è pronto, ritorna None.
        """
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            depends_on = getattr(task, "depends_on", []) or []
            ready = True
            for dep_id in depends_on:
                dep = next((t for t in self.tasks if t.id == dep_id), None)
                if dep is None:
                    # dipendenza sconosciuta → non blocchiamo il task
                    continue
                if dep.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    # dipendenza non ancora completata
                    ready = False
                    break
            if ready:
                return task

        return None

    def has_pending_tasks(self) -> bool:
        return any(task.status == TaskStatus.PENDING for task in self.tasks)


@dataclass
class EmotionDelta:
    curiosity: float = 0.0
    fatigue: float = 0.0
    frustration: float = 0.0
    confidence: float = 0.0

    # nuovo: nucleo affettivo + drive
    mood: float = 0.0            # [-1..1] -> negativo / positivo
    energy: float = 0.0          # [0..1]

    playfulness: float = 0.0     # [0..1] voglia di giocare
    social_need: float = 0.0     # [0..1] bisogno di contatto con l’utente
    learning_drive: float = 0.0  # [0..1] voglia di imparare / esplorare
    
@dataclass
class EmotionalState:
    # core attuale
    curiosity: float = 0.5
    fatigue: float = 0.0
    frustration: float = 0.0
    confidence: float = 0.5

    # nucleo aurion-like
    mood: float = 0.0        # [-1..1] 0 = neutro, >0 positivo, <0 negativo
    energy: float = 0.6      # [0..1] livello di attivazione

    # drive “psicologici”
    playfulness: float = 0.3
    social_need: float = 0.4
    learning_drive: float = 0.7

    def apply_delta(self, delta: EmotionDelta) -> None:
        # 4 slider originali
        self.curiosity = self._clamp01(self.curiosity + delta.curiosity)
        self.fatigue = self._clamp01(self.fatigue + delta.fatigue)
        self.frustration = self._clamp01(self.frustration + delta.frustration)
        self.confidence = self._clamp01(self.confidence + delta.confidence)

        # mood/energy tipo Aurion
        self.mood = self._clamp(self.mood + delta.mood, -1.0, 1.0)
        self.energy = self._clamp01(self.energy + delta.energy)

        # drive
        self.playfulness = self._clamp01(self.playfulness + delta.playfulness)
        self.social_need = self._clamp01(self.social_need + delta.social_need)
        self.learning_drive = self._clamp01(self.learning_drive + delta.learning_drive)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

class MemoryScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    GLOBAL = "global"
    CONVERSATION = "conversation"


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class AgentRunStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class MemoryKeys:
    # Profili utente
    USER_PROFILE_PREFIX = "user_profile:"

    # Contesto progetto
    PROJECT_CONTEXT_PREFIX = "project_context::"

    # Risultati R
    R_EDA_RESULT = "r_eda_result"
    R_MODELING_RESULT = "r_modeling_result"
    R_REPORT_RESULT = "r_report_result"

    # Varie
    DATABASE_SCHEMA = "database_schema"

@dataclass
class MemoryItem:
    id: str
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentRun:
    id: str
    agent_name: str
    input_payload: Dict[str, Any]
    output_payload: Dict[str, Any]
    status: AgentRunStatus
    emotion_delta: EmotionDelta = field(default_factory=EmotionDelta)
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationContext:
    id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    plan: Optional[Plan] = None
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    
    def add_message(self, role: MessageRole, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.utcnow()


def new_id() -> str:
    import uuid
    return str(uuid.uuid4())

class EventType(str, Enum):
    REQUEST_RECEIVED = "REQUEST_RECEIVED"
    PLAN_CREATED = "PLAN_CREATED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    AGENT_RUN_COMPLETED = "AGENT_RUN_COMPLETED"
    AGENT_RUN_FAILED = "AGENT_RUN_FAILED"


@dataclass
class Event:
    id: str
    type: EventType
    correlation_id: str
    timestamp: datetime
    payload: Dict[str, Any] = field(default_factory=dict)