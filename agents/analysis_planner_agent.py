from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    MemoryScope,
    MemoryType,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class AnalysisPlannerAgent(Agent):
    name = "analysis_planner_agent"
    description = (
        "Progetta un piano di analisi dati generico (classification / regression / time-series / exploratory) "
        "usando, se disponibile, la scheda requisiti salvata dal RequirementsAgent."
    )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _load_requirements_sheet(
        context: ConversationContext,
        memory: MemoryEngine,
    ) -> Optional[Dict[str, Any]]:
        """
        Prova a caricare la requirements_sheet salvata dal RequirementsAgent
        a livello di conversazione e, se presente, a livello di progetto.
        """
        # 1) Scope CONVERSATION
        conv_id = getattr(context, "id", None)
        if conv_id:
            key_conv = f"requirements_sheet:{conv_id}"
            try:
                raw = memory.load_item_content(
                    key=key_conv,
                    scope=MemoryScope.CONVERSATION,
                    type_=MemoryType.PROCEDURAL,
                )
                if raw:
                    return json.loads(raw)
            except Exception:
                pass

        # 2) Scope PROJECT (se abbiamo un project_id)
        project_id = getattr(context, "project_id", None) or getattr(
            context, "current_project_id", None
        )
        if project_id:
            key_proj = f"requirements_sheet:{project_id}"
            try:
                raw = memory.load_item_content(
                    key=key_proj,
                    scope=MemoryScope.PROJECT,
                    type_=MemoryType.PROCEDURAL,
                )
                if raw:
                    return json.loads(raw)
            except Exception:
                pass

        return None

    # ---------------------------------------------------------
    # Core
    # ---------------------------------------------------------

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,          # noqa: ARG002
        emotional_state: EmotionalState,  # noqa: ARG002
    ) -> AgentResult:
        # 1) Carichiamo eventuale scheda requisiti
        req_sheet = self._load_requirements_sheet(context, memory)

        # 2) Determiniamo il problem_type
        problem_type = input_payload.get("problem_type")
        source = "input_payload"
        if not problem_type and req_sheet:
            problem_type = (req_sheet.get("problem_type") or "").lower() or None
            source = "requirements_sheet"

        # normalizziamo
        if problem_type not in {"classification", "regression", "time-series"}:
            # se la scheda parla di clustering/exploratory/other → trattiamo come exploratory
            if problem_type in {"clustering", "exploratory", "other"}:
                problem_type = "exploratory"
            else:
                # default se non specificato
                problem_type = "classification"

        # 3) Raccogliamo info principali dal requirements_sheet (se c'è)
        primary_goal = None
        target_variable = None
        domain = None
        constraints: Dict[str, Any] = {}
        evaluation: Dict[str, Any] = {}

        if req_sheet:
            primary_goal = req_sheet.get("primary_goal")
            target_variable = req_sheet.get("target_variable")
            domain = req_sheet.get("domain")
            constraints = req_sheet.get("constraints") or {}
            evaluation = req_sheet.get("evaluation") or {}

        # 3.b) Data type (per ora default 'tabular', ma estendibile)
        data_type = "tabular"
        if input_payload.get("data_type"):
            data_type = str(input_payload["data_type"])

        # 4) Costruiamo i passi del piano in base al tipo di problema
        steps: List[str]

        # Testo extra da inserire nei punti 1–2 se abbiamo info
        goal_txt = f" (obiettivo: {primary_goal})" if primary_goal else ""
        tgt_txt = (
            f" (target: {target_variable})" if target_variable else ""
        )

        if problem_type == "regression":
            steps = [
                f"1. Esplorazione iniziale del dataset{goal_txt}: dimensioni, percentuale di valori mancanti, distribuzioni delle principali variabili.",
                f"2. Definizione chiara della variabile target continua{tgt_txt} e delle feature candidate.",
                "3. Pulizia dati e feature engineering (scaling, trasformazioni, interazioni, gestione outlier).",
                "4. Train/test split o cross-validation, in funzione della dimensione del dataset e dei vincoli temporali.",
                "5. Addestramento di modelli di regressione (lineare, elastic net, modelli ad alberi/gradient boosting).",
                "6. Valutazione con RMSE, MAE, R² e confronto tra modelli.",
                "7. Analisi di interpretabilità (coefficenti, feature importance, partial dependence) e sintesi business-friendly.",
            ]
        elif problem_type == "time-series":
            steps = [
                f"1. Analisi preliminare della serie temporale{goal_txt}: trend, stagionalità, outlier e cambi di regime.",
                "2. Definizione dell'orizzonte di previsione e della granularità (giornaliera, settimanale, mensile, ecc.).",
                "3. Creazione di variabili lag, rolling statistics, indicatori di calendario e eventuali covariate esterne.",
                "4. Split temporale train/validation/test, rispettando l'ordine cronologico.",
                "5. Addestramento di modelli ARIMA/ETS/Prophet o modelli ML con feature temporali.",
                "6. Valutazione su finestre temporali con MAPE, sMAPE, RMSE e confronto tra approcci.",
                "7. Analisi dei residui, diagnostica del modello e definizione di una strategia di aggiornamento periodico.",
            ]
        elif problem_type == "exploratory":
            steps = [
                f"1. Comprensione del contesto e degli obiettivi esplorativi{goal_txt} (quali domande vogliamo davvero porre ai dati).",
                "2. Analisi strutturale del dataset: dimensioni, tipi di variabili, percentuale di NA.",
                "3. Esplorazione univariata e bivariata delle variabili chiave (distribuzioni, boxplot, correlazioni).",
                "4. Identificazione di pattern, segmenti interessanti e possibili anomalie.",
                "5. Se utile, applicazione di tecniche di riduzione di dimensionalità o clustering esplorativo.",
                "6. Sintesi visuale (grafici, tabelle) e definizione di ipotesi/idee per eventuale modellazione successiva.",
            ]
        else:  # classification (default, compatibile con churn)
            steps = [
                f"1. Esplorazione iniziale del dataset{goal_txt}: dimensioni, qualità dei dati, distribuzioni di feature e target.",
                f"2. Definizione chiara della variabile target e della finestra temporale{tgt_txt}.",
                "3. Feature engineering specifica (es. recency/frequency/importi per clienti, variabili di comportamento, canali).",
                "4. Train/test split (holdout o coorte temporale) con attenzione a bilanciamento e leakage.",
                "5. Addestramento di uno o più modelli di classificazione (logistica, random forest, gradient boosting, ecc.).",
                "6. Valutazione con AUC, precision/recall, confusion matrix, curve di lift/gain.",
                "7. Analisi di interpretabilità (feature importance, partial dependence, SHAP) e raccomandazioni operative.",
            ]

        # 4.b) Recommended agents per pipeline: EDA → Modeling → Explainability
        recommended_agents: List[str] = []

        # Sempre EDA in R come primo passo
        recommended_agents.append("r_eda_agent")

        # Per classification/regression usiamo r_analysis_agent come demo/modeling
        if problem_type in {"classification", "regression"}:
            recommended_agents.append("r_analysis_agent")

        # Chiudiamo con ExplanationAgent per la parte narrativa
        recommended_agents.append("explanation_agent")

        # 5) Costruiamo un oggetto piano strutturato da salvare in memoria
        plan_struct: Dict[str, Any] = {
            "schema_version": 1,
            "source": source,
            "problem_type": problem_type,
            "primary_goal": primary_goal,
            "target_variable": target_variable,
            "domain": domain,
            "constraints": constraints,
            "evaluation": evaluation,
            # nuovo: tipo di dato principale (per ora tabellare)
            "data_type": data_type,
            # nuovo: alias esplicito, utile per Explanation / altri agent
            "analysis_steps": steps,
            # manteniamo anche il campo originale 'steps' per retrocompatibilità
            "steps": steps,
            # nuovo: pipeline consigliata di agent da chiamare
            "recommended_agents": recommended_agents,
        }

        # 6) Salviamo in memoria
        try:
            memory.store_item(
                scope=MemoryScope.CONVERSATION,
                type_=MemoryType.PROCEDURAL,
                key="analysis_plan_text",
                content="\n".join(steps),
                metadata={"agent": self.name},
            )
        except Exception:
            pass

        # 7) Output: niente testo per l'utente, lo farà ExplanationAgent
        output = {
            "user_visible_message": "",
            # versioni “umane”
            "analysis_plan": steps,
            "problem_type": problem_type,
            "requirements_used": bool(req_sheet),
            # versione strutturata per agent cognitivi / Explanation
            "analysis_plan_structured": plan_struct,
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(confidence=0.05)
        return AgentResult(output_payload=output, emotion_delta=delta)
