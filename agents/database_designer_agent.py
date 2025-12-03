from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List

from core.agents_base import Agent, AgentResult
from core.models import EmotionalState, EmotionDelta, MemoryKeys, ConversationContext
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class DatabaseDesignerAgent(Agent):
    name = "database_designer_agent"
    description = "Disegna uno schema semplice e crea un database SQLite sul disco."

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        user_message = input_payload.get("user_message", "").lower()

        # euristica minimale: se l'utente parla di clienti/ordini, proponiamo due tabelle
        tables_sql: List[str] = []

        if "cliente" in user_message or "clienti" in user_message:
            tables_sql.append(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TEXT
                );
                """.strip()
            )

        if "ordini" in user_message or "ordine" in user_message or "orders" in user_message:
            tables_sql.append(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    order_date TEXT,
                    amount REAL,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                );
                """.strip()
            )

        # se non riconosciamo nulla, creiamo una tabella generica
        if not tables_sql:
            tables_sql.append(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT
                );
                """.strip()
            )

        # path del DB da creare/aggiornare
        db_path = os.path.join(os.getcwd(), "designed_app.db")

        executed_sql = "\n\n".join(tables_sql)

        # creiamo fisicamente il DB e le tabelle
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            for stmt in tables_sql:
                cur.execute(stmt)
            conn.commit()
            conn.close()
            success = True
            error_msg = ""
        except Exception as exc:  # noqa: BLE001
            success = False
            error_msg = str(exc)

        if success:
            text = (
                "Ho progettato e creato (o aggiornato) un database SQLite.\n\n"
                f"Percorso del file: `{db_path}`\n\n"
                "Ho eseguito queste istruzioni SQL:\n\n"
                f"{executed_sql}\n\n"
                "Puoi aprire il file con qualsiasi tool SQLite."
            )
            status_delta = EmotionDelta(confidence=0.08, curiosity=0.03)
            output = {
                "user_visible_message": "",  # lascio che sia l'explanation_agent a parlare
                "db_sql": executed_sql,
                "db_path": db_path,
                "stop_for_user_input": False,
            }
        else:
            text = (
                "Ho provato a creare il database, ma ho incontrato un errore:\n\n"
                f"{error_msg}"
            )
            status_delta = EmotionDelta(confidence=-0.05, frustration=0.1)
            output = {
                "user_visible_message": text,
                "db_sql": "",
                "db_path": db_path,
                "stop_for_user_input": False,
            }

        # opzionale: memorizziamo lo schema in memoria procedurale
        try:
            from core.models import MemoryScope, MemoryType  # type: ignore

            memory.store_item(
                scope=MemoryScope.CONVERSATION,
                type_=MemoryType.PROCEDURAL,
                key=MemoryKeys.DATABASE_SCHEMA,
                content=executed_sql,
                metadata={"db_path": db_path},
            )
        except Exception:
            pass

        return AgentResult(output_payload=output, emotion_delta=status_delta)
