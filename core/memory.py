from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    MemoryItem,
    MemoryScope,
    MemoryType,
    MemoryKeys,
    Message,
    MessageRole,
    AgentRun,
    AgentRunStatus,
    EmotionDelta,
    new_id,
)


class MemoryEngine:
    """
    Motore di memoria persistente su SQLite.
    - messages: log dei messaggi
    - memory_items: memoria episodica/semantica/procedurale
    - agent_runs: log delle esecuzioni degli agent
    - agent_definitions: definizioni di agent generate dalla pipeline
    - events: event log (REQUEST / PLAN / TASK / RUN, ecc.)
    """

    def __init__(self, db_path: str = "cognitive_memory.db") -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_recent_agent_runs(self, limit: int = 50) -> List[AgentRun]:
        """
        Ritorna gli ultimi `limit` AgentRun dal DB, in ordine cronologico crescente.
        Utile per DiagnosticsAgent, replay, audit, ecc.
        """
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                agent_name,
                input_json,
                output_json,
                status,
                curiosity,
                fatigue,
                frustration,
                confidence,
                started_at,
                finished_at
            FROM agent_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        runs: List[AgentRun] = []
        for (
            run_id,
            agent_name,
            input_json,
            output_json,
            status_str,
            curiosity,
            fatigue,
            frustration,
            confidence,
            started_at_str,
            finished_at_str,
        ) in rows:
            runs.append(
                AgentRun(
                    id=run_id,
                    agent_name=agent_name,
                    input_payload=json.loads(input_json),
                    output_payload=json.loads(output_json),
                    status=AgentRunStatus(status_str),
                    emotion_delta=EmotionDelta(
                        curiosity=curiosity or 0.0,
                        fatigue=fatigue or 0.0,
                        frustration=frustration or 0.0,
                        confidence=confidence or 0.0,
                    ),
                    started_at=datetime.fromisoformat(started_at_str),
                    finished_at=datetime.fromisoformat(finished_at_str),
                )
            )

        # Restituiamo dal più vecchio al più nuovo
        return list(reversed(runs))

    # ----------------- Metriche agent (da DiagnosticsAgent) -----------------

    def get_last_diagnostics(self) -> Optional[Dict[str, Any]]:
        """
        Ritorna l'ultimo payload 'diagnostics' prodotto da diagnostics_agent,
        se esiste, altrimenti None.
        """
        try:
            runs = self.get_recent_agent_runs(limit=200)
        except Exception:
            return None

        for r in reversed(runs):
            if r.agent_name == "diagnostics_agent":
                diag = r.output_payload.get("diagnostics")
                if isinstance(diag, dict):
                    return diag
                break
        return None

    def get_agent_metrics_from_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """
        Usa l'ultimo diagnostics per costruire una mappa:
        { agent_name: { failure_rate, total_runs, avg_duration, global_avg_duration } }
        """
        diag = self.get_last_diagnostics()
        if not diag:
            return {}

        metrics: Dict[str, Dict[str, Any]] = {}

        failures = diag.get("failures") or []
        for item in failures:
            name = item.get("agent_name")
            if not name:
                continue
            m = metrics.setdefault(name, {})
            m["failure_rate"] = float(item.get("failure_rate") or 0.0)
            m["total_runs"] = int(item.get("total_runs") or 0)

        perf = diag.get("performance") or {}
        global_avg = float(perf.get("global_avg") or 0.0)
        slow_agents = perf.get("slow_agents") or []
        for s in slow_agents:
            name = s.get("agent_name")
            if not name:
                continue
            m = metrics.setdefault(name, {})
            m["avg_duration"] = float(s.get("avg_duration") or 0.0)
            m["global_avg_duration"] = global_avg

        return metrics

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        cur = conn.cursor()
        # Schema base + tabella events
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                type TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                status TEXT NOT NULL,
                curiosity REAL,
                fatigue REAL,
                frustration REAL,
                confidence REAL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_definitions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                parent_id TEXT,
                lifecycle_state TEXT NOT NULL DEFAULT 'draft'
            );

            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )

        cur.executescript(
            """
            -- Indici per memory_items
            CREATE INDEX IF NOT EXISTS idx_memory_items_scope_type_key
              ON memory_items(scope, type, key);

            CREATE INDEX IF NOT EXISTS idx_memory_items_scope_type_created
              ON memory_items(scope, type, created_at);
            """
        )

        # Migrazione soft: se la tabella agent_definitions esisteva senza lifecycle_state
        try:
            cur.execute(
                "ALTER TABLE agent_definitions "
                "ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'draft'"
            )
        except sqlite3.OperationalError:
            # colonna già esistente → ignoriamo
            pass

        conn.commit()
        conn.close()

    # ----------------- Memoria items ---------------------------------

    def store_item(
        self,
        scope: MemoryScope,
        type_: MemoryType,
        key: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        # Normalizza content a stringa
        if isinstance(content, str):
            content_str = content
        else:
            try:
                content_str = json.dumps(content, ensure_ascii=False)
            except Exception:
                content_str = str(content)

        item = MemoryItem(
            id=new_id(),
            scope=scope,
            type=type_,
            key=key,
            content=content_str,
            metadata=metadata or {},
        )
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO memory_items (id, scope, type, key, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.scope.value,
                item.type.value,
                item.key,
                item.content,
                json.dumps(item.metadata),
                item.created_at.isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        return item

    def search_items(
        self,
        scope: Optional[MemoryScope] = None,
        type_: Optional[MemoryType] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryItem]:
        sql = """
            SELECT id, scope, type, key, content, metadata_json, created_at
            FROM memory_items
        """
        clauses: List[str] = []
        params: List[Any] = []

        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope.value)
        if type_ is not None:
            clauses.append("type = ?")
            params.append(type_.value)
        if query is not None:
            clauses.append("content LIKE ?")
            params.append(f"%{query}%")

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        items: List[MemoryItem] = []
        for (
            item_id,
            scope_str,
            type_str,
            key,
            content,
            metadata_json,
            created_at_str,
        ) in rows:
            items.append(
                MemoryItem(
                    id=item_id,
                    scope=MemoryScope(scope_str),
                    type=MemoryType(type_str),
                    key=key,
                    content=content,
                    metadata=json.loads(metadata_json),
                    created_at=datetime.fromisoformat(created_at_str),
                )
            )
        return items

    def find_items_by_key(
        self,
        key: str,
        scope: Optional[MemoryScope] = None,
        type_: Optional[MemoryType] = None,
        limit: int = 50,
    ) -> List[MemoryItem]:
        """
        Ritorna gli ultimi MemoryItem con una certa key, opzionalmente filtrando
        per scope/type. Utile per alert strutturati (critic_suggestion, security_alert, ecc.).
        """
        sql = """
            SELECT id, scope, type, key, content, metadata_json, created_at
            FROM memory_items
            WHERE key = ?
        """
        clauses: List[str] = []
        params: List[Any] = [key]

        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope.value)
        if type_ is not None:
            clauses.append("type = ?")
            params.append(type_.value)

        if clauses:
            sql += " AND " + " AND ".join(clauses)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        items: List[MemoryItem] = []
        for (
            item_id,
            scope_str,
            type_str,
            key,
            content,
            metadata_json,
            created_at_str,
        ) in rows:
            items.append(
                MemoryItem(
                    id=item_id,
                    scope=MemoryScope(scope_str),
                    type=MemoryType(type_str),
                    key=key,
                    content=content,
                    metadata=json.loads(metadata_json),
                    created_at=datetime.fromisoformat(created_at_str),
                )
            )
        return items

    def load_item_content(
        self,
        key: str,
        scope: Optional[MemoryScope] = None,
        type_: Optional[MemoryType] = None,
    ) -> Optional[str]:
        """
        Ritorna il contenuto dell'ultimo MemoryItem con quella key
        (e, opzionalmente, scope/type), oppure None se non trovato.
        Utile per agent che vogliono recuperare velocemente
        un risultato precedente (es. r_eda_result, r_modeling_result, ecc.).
        """
        sql = """
            SELECT content
            FROM memory_items
        """
        clauses: List[str] = ["key = ?"]
        params: List[Any] = [key]

        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope.value)
        if type_ is not None:
            clauses.append("type = ?")
            params.append(type_.value)

        sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT 1"

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        conn.close()

        if row is None:
            return None
        return row[0]

    def load_user_profile_json(self, user_id: str) -> Optional[str]:
        """
        Ritorna il contenuto JSON dell'ultimo profilo utente per user_id,
        oppure None se non esiste ancora.

        Convenzione fissa:
          - scope = MemoryScope.USER
          - type_ = MemoryType.SEMANTIC
          - key   = f"user_profile:{user_id}"

        NOTA: questo metodo NON fa parsing JSON, restituisce solo la stringa.
        Saranno gli agent (es. user_profile_agent) a fare json.loads(...) e
        a gestire schema_version & co.
        """
        return self.load_item_content(
            key=f"{MemoryKeys.USER_PROFILE_PREFIX}{user_id}",
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
        )

    def save_user_profile_json(
        self,
        user_id: str,
        content_json: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        Salva un nuovo snapshot del profilo utente (JSON) per user_id.

        - Verifica che content_json sia una stringa JSON valida.
        - Usa sempre:
            scope = USER
            type_ = SEMANTIC
            key   = f"user_profile:{user_id}"

        Non applica logiche di merge: ogni chiamata crea un nuovo MemoryItem.
        Sarà il client (es. user_profile_agent) a produrre il JSON completo.
        """
        # Validazione soft: deve essere JSON valido
        try:
            parsed = json.loads(content_json)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"save_user_profile_json: content_json non è JSON valido: {exc}") from exc

        # opzionale: controllino che user_id dentro al JSON coincida
        profile_user_id = parsed.get("user_id")
        if isinstance(profile_user_id, str) and profile_user_id != user_id:
            # non blocchiamo, ma segnaliamo nei metadata
            meta = metadata or {}
            meta.setdefault("warnings", [])
            meta["warnings"].append(
                f"user_id nel JSON ('{profile_user_id}') diverso da parametro ('{user_id}')"
            )
            metadata = meta

        return self.store_item(
            scope=MemoryScope.USER,
            type_=MemoryType.SEMANTIC,
            key=f"{MemoryKeys.USER_PROFILE_PREFIX}{user_id}",
            content=content_json,
            metadata=metadata or {},
        )

    # ----------------- Log messaggi ----------------------------------

    def log_message(self, message: Message) -> None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (role, content, ts)
            VALUES (?, ?, ?)
            """,
            (message.role.value, message.content, message.timestamp.isoformat()),
        )
        conn.commit()
        conn.close()

    def get_recent_messages(self, limit: int = 20) -> List[Message]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content, ts
            FROM messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        messages: List[Message] = []
        # invertiamo per avere ordine cronologico
        for role_str, content, ts in reversed(rows):
            messages.append(
                Message(
                    role=MessageRole(role_str),
                    content=content,
                    timestamp=datetime.fromisoformat(ts),
                )
            )
        return messages

    # ----------------- Log esecuzioni agent --------------------------

    def log_agent_run(self, run: AgentRun) -> None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO agent_runs (
                id, agent_name, input_json, output_json, status,
                curiosity, fatigue, frustration, confidence,
                started_at, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.agent_name,
                json.dumps(run.input_payload),
                json.dumps(run.output_payload),
                run.status.value,
                run.emotion_delta.curiosity,
                run.emotion_delta.fatigue,
                run.emotion_delta.frustration,
                run.emotion_delta.confidence,
                run.started_at.isoformat(),
                run.finished_at.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    # ----------------- Definizioni di agent --------------------------

    def save_agent_definition(self, definition: Dict[str, Any]) -> None:
        """
        Salva/aggiorna una AgentDefinition logica (come dict) nel DB.
        Chiamiamo questo dalla pipeline Architect/Validator/Critic.
        Il dict atteso contiene almeno:
          - id, name, description
          - config (dict)
          - created_at (datetime) opzionale
          - is_active (bool) opzionale
          - parent_id opzionale
          - lifecycle_state opzionale (default: draft)
        """
        conn = self._get_conn()
        cur = conn.cursor()

        config = definition.get("config", {})
        created_at = definition.get("created_at", datetime.utcnow())
        lifecycle_state = definition.get("lifecycle_state", "draft") or "draft"

        cur.execute(
            """
            INSERT INTO agent_definitions (
                id, name, description, config_json, created_at,
                is_active, parent_id, lifecycle_state
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                config_json = excluded.config_json,
                is_active = excluded.is_active,
                parent_id = excluded.parent_id,
                lifecycle_state = excluded.lifecycle_state
            """,
            (
                definition["id"],
                definition["name"],
                definition.get("description", ""),
                json.dumps(config),
                created_at.isoformat(),
                1 if definition.get("is_active", False) else 0,
                definition.get("parent_id"),
                lifecycle_state,
            ),
        )
        conn.commit()
        conn.close()

    def list_agent_definitions(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, description, config_json,
                   created_at, is_active, parent_id, lifecycle_state
            FROM agent_definitions
            ORDER BY created_at ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        defs: List[Dict[str, Any]] = []
        for (
            id_,
            name,
            description,
            config_json,
            created_at_str,
            is_active_int,
            parent_id,
            lifecycle_state,
        ) in rows:
            defs.append(
                {
                    "id": id_,
                    "name": name,
                    "description": description,
                    "config": json.loads(config_json),
                    "created_at": datetime.fromisoformat(created_at_str),
                    "is_active": bool(is_active_int),
                    "parent_id": parent_id,
                    "lifecycle_state": lifecycle_state or "draft",
                }
            )
        return defs

    # ----------------- Event log -------------------------------------

    def log_event(
        self,
        type_: EventType,
        correlation_id: str,
        payload: Dict[str, Any],
    ) -> Event:
        """
        Logga un evento atomico (REQUEST_RECEIVED, PLAN_CREATED, ecc.) nel DB.
        Ritorna l'Event creato (utile per debug e test).
        """
        ev = Event(
            id=new_id(),
            type=type_,
            correlation_id=correlation_id,
            timestamp=datetime.utcnow(),
            payload=payload,
        )

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (id, type, correlation_id, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ev.id,
                ev.type.value,
                ev.correlation_id,
                ev.timestamp.isoformat(),
                json.dumps(ev.payload),
            ),
        )
        conn.commit()
        conn.close()
        return ev

    def get_events(
        self,
        correlation_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Event]:
        """
        Ritorna gli ultimi eventi, opzionalmente filtrati per correlation_id.
        Utile per DiagnosticsAgent, replay, audit.
        """
        sql = """
            SELECT id, type, correlation_id, timestamp, payload_json
            FROM events
        """
        params: List[Any] = []
        if correlation_id:
            sql += " WHERE correlation_id = ?"
            params.append(correlation_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        events: List[Event] = []
        for ev_id, type_str, cid, ts_str, payload_json in rows:
            events.append(
                Event(
                    id=ev_id,
                    type=EventType(type_str),
                    correlation_id=cid,
                    timestamp=datetime.fromisoformat(ts_str),
                    payload=json.loads(payload_json),
                )
            )
        # restituiamo in ordine cronologico crescente
        return list(reversed(events))


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
