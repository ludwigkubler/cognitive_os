# agents/explanation_agent.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class ExplanationAgent(Agent):
    name = "explanation_agent"
    description = (
        "Spiega in modo umano i risultati degli altri agent (R churn demo, "
        "EDA/modeling, schema DB) adattando il livello di dettaglio al profilo utente."
    )

    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:
        # --- 1) recupero piano di analisi (AnalysisPlannerAgent) ---
        analysis_plan: List[str] = []
        if context.plan is not None:
            for task in context.plan.tasks:
                if (
                    task.agent_name == "analysis_planner_agent"
                    and task.result
                    and "analysis_plan" in task.result
                ):
                    analysis_plan = (
                        task.result.get("analysis_plan") or []
                        if isinstance(task.result.get("analysis_plan"), list)
                        else []
                    )

        # --- 2) recupero risultati R churn demo (RAnalysisAgent) ---
        r_churn_result: Optional[Dict[str, Any]] = None
        if context.plan is not None:
            for task in context.plan.tasks:
                if (
                    task.agent_name == "r_analysis_agent"
                    and task.result
                    and "r_result" in task.result
                ):
                    # prendo l'ultimo risultato disponibile
                    r_churn_result = task.result.get("r_result")

        # --- 3) recupero risultati EDA generica (REdaAgent / script eda_generic.R) ---
        r_eda_result: Optional[Dict[str, Any]] = None
        if context.plan is not None:
            for task in context.plan.tasks:
                if (
                    task.agent_name == "r_eda_agent"
                    and task.result
                    and "r_eda_result" in task.result
                ):
                    r_eda_result = task.result.get("r_eda_result")

        # --- 4) risultati eventuale modellazione generica (ipotetico RModelingAgent) ---
        modeling_result: Optional[Dict[str, Any]] = None
        if context.plan is not None:
            for task in context.plan.tasks:
                # lascio spazio a naming futuri, senza rompere nulla se non esistono
                if (
                    task.agent_name in ("r_modeling_agent", "modeling_agent")
                    and task.result
                ):
                    # accetto sia "modeling_result" che "r_modeling_result"
                    modeling_result = (
                        task.result.get("modeling_result")
                        or task.result.get("r_modeling_result")
                        or task.result.get("modeling")
                    )

        # --- 5) eventuale schema DB (DatabaseDesignerAgent) ---
        db_schema: Optional[str] = None
        if context.plan is not None:
            for task in context.plan.tasks:
                if task.agent_name == "database_designer_agent" and task.result:
                    if "ddl_sql" in task.result:
                        db_schema = task.result["ddl_sql"]
                    elif "db_sql" in task.result:
                        db_schema = task.result["db_sql"]

        # --- 6) profilazione utente dal MemoryEngine (per tarare dettaglio) ---
        detail_level = "medium"  # default
        profile = None
        try:
            profile = memory.load_user_profile_json(context.user_id)
        except Exception:
            profile = None

        if isinstance(profile, dict):
            prefs = profile.get("conversational_preferences") or {}
            # gioco con 3 livelli se presenti, altrimenti rimane medium
            level = prefs.get("detail_level")
            if level in {"low", "medium", "high"}:
                detail_level = level
            else:
                # fallback molto semplice su un paio di flag possibili
                if prefs.get("prefers_concise") is True:
                    detail_level = "low"
                elif prefs.get("prefers_detailed") is True:
                    detail_level = "high"

        parts: List[str] = []

        # piccolo header adattato al profilo
        if detail_level == "low":
            parts.append("Ti faccio un riassunto rapido di quello che ho fatto finora:")
        elif detail_level == "high":
            parts.append(
                "Ti racconto in modo un po' più tecnico e dettagliato cosa ho fatto finora:"
            )
        else:
            parts.append("Riassumo i passi principali che ho eseguito finora:")

        parts.append("")

        # --- 7) se presenti: EDA generica ---
        if r_eda_result is not None:
            ok_flag = r_eda_result.get("ok")
            if ok_flag is False:
                parts.append(
                    "1) Analisi esplorativa dei dati (EDA in R) – NON riuscita:"
                )
                err = r_eda_result.get("error") or "Errore sconosciuto."
                parts.append(f"   · Lo script EDA ha segnalato: {err}")
                parts.append("")
            else:
                eda = r_eda_result.get("eda") or {}
                n_rows = eda.get("n_rows")
                n_cols = eda.get("n_cols")

                parts.append("1) Analisi esplorativa dei dati (EDA in R):")
                if n_rows is not None and n_cols is not None:
                    parts.append(
                        f"   · Il dataset ha circa {n_rows} righe e {n_cols} colonne."
                    )

                missing = eda.get("missing_perc") or {}
                if isinstance(missing, dict) and missing:
                    # prendo le colonne con NA > 0 e ordino decrescente
                    miss_items = [
                        (name, float(val))
                        for name, val in missing.items()
                        if val is not None
                    ]
                    miss_items.sort(key=lambda x: x[1], reverse=True)
                    if miss_items:
                        top = miss_items[:3]
                        descr = ", ".join(
                            f"{name} (~{val:.1f}% NA)" for name, val in top
                        )
                        parts.append(
                            f"   · Alcune colonne hanno valori mancanti importanti: {descr}."
                        )

                if detail_level == "high":
                    num_summary = eda.get("numeric_summary") or {}
                    if isinstance(num_summary, dict) and num_summary:
                        parts.append(
                            "   · Per alcune variabili numeriche ho stimato media/mediana "
                            "e range (min–max)."
                        )
                    corr = eda.get("numeric_corr_head")
                    if corr is not None:
                        parts.append(
                            "   · Ho anche calcolato una matrice di correlazione (parziale) "
                            "tra le variabili numeriche."
                        )

                parts.append("")

        # --- 8) modellazione generica (regressione / classificazione) ---
        if modeling_result is not None:
            # qui assumo struttura compatibile con modeling_generic.R
            mt = modeling_result.get("model_type") or "sconosciuto"
            n_obs = modeling_result.get("n_obs")
            train_size = modeling_result.get("train_size")
            test_size = modeling_result.get("test_size")

            parts.append("2) Modellazione predittiva:")
            desc_base = f"   · Ho addestrato un modello di tipo {mt}"
            if n_obs is not None:
                desc_base += f" su circa {n_obs} osservazioni"
            desc_base += "."
            parts.append(desc_base)

            if train_size is not None and test_size is not None:
                parts.append(
                    f"   · Split train/test: {train_size} righe per l'addestramento, "
                    f"{test_size} per la valutazione."
                )

            acc = modeling_result.get("accuracy")
            rmse = modeling_result.get("rmse")
            mae = modeling_result.get("mae")
            r2 = modeling_result.get("r2")

            if acc is not None:
                parts.append(f"   · Accuratezza sul test: {acc:.3f}.")
            if rmse is not None:
                parts.append(f"   · RMSE sul test: {rmse:.3f}.")
            if mae is not None:
                parts.append(f"   · MAE sul test: {mae:.3f}.")
            if r2 is not None:
                parts.append(f"   · R² sul test: {r2:.3f}.")

            if detail_level == "high":
                coefs = modeling_result.get("coefficients") or {}
                if isinstance(coefs, dict) and coefs:
                    parts.append(
                        "   · Ho anche stimato i coefficienti del modello "
                        "(peso delle varie feature sul target)."
                    )

            parts.append("")

        # --- 9) demo churn/logistic in R ---
        if r_churn_result is not None:
            ok_flag = r_churn_result.get("ok", True)
            if ok_flag is False:
                parts.append("3) Demo di churn in R – NON riuscita:")
                err = r_churn_result.get("error") or "Errore sconosciuto."
                parts.append(f"   · Lo script di churn ha segnalato: {err}")
                parts.append("")
            else:
                n = r_churn_result.get("n")
                churn_rate = r_churn_result.get("churn_rate")

                parts.append("3) Demo di churn (logistica in R):")
                if n is not None and churn_rate is not None:
                    parts.append(
                        f"   · Ho simulato un dataset sintetico con circa {n} clienti."
                    )
                    parts.append(
                        f"   · Nel campione, il tasso di churn simulato è ~{churn_rate:.2%}."
                    )

                coefs = r_churn_result.get("coefficients") or []
                if detail_level != "low" and isinstance(coefs, list) and coefs:
                    parts.append("   · Coefficienti principali del modello logit:")
                    # mostro solo qualche coefficiente se sono tanti
                    for c in coefs[:5]:
                        term = c.get("term")
                        est = c.get("estimate")
                        pval = c.get("p_value")
                        if term is None or est is None:
                            continue
                        if pval is not None:
                            parts.append(
                                f"      - {term}: stima={est:.3f}, p-value={pval:.3f}"
                            )
                        else:
                            parts.append(f"      - {term}: stima={est:.3f}")

                parts.append("")

        # --- 10) schema DB, se presente ---
        if db_schema:
            parts.append("4) Progettazione schema di database (DDL SQL):")
            if detail_level == "low":
                parts.append(
                    "   · Ho generato uno schema SQL compatibile con i dati e il flusso analitico."
                )
            else:
                parts.append(
                    "   · Ho prodotto uno schema SQL (DDL) che può essere usato per "
                    "persistenza e orchestrazione delle analisi."
                )
            if detail_level == "high":
                parts.append("")
                parts.append("Snippet DDL (estratto):")
                # non faccio slicing aggressivo: ti lascio eventualmente gestire tu il trunc
                parts.append(db_schema)

            parts.append("")

        # --- 11) Se ho un piano di analisi teorico, lo racconto ---
        if analysis_plan:
            if any([r_eda_result, modeling_result, r_churn_result]):
                parts.append(
                    "Infine, ho anche esplicitato un flusso di lavoro generale per le prossime azioni:"
                )
            else:
                parts.append(
                    "Ti propongo questo flusso di lavoro generale per affrontare l'analisi:"
                )
            for step in analysis_plan:
                parts.append(f"   · {step}")
            parts.append("")

        # --- 12) Meta-spiegazione sullo stato interno (breve) ---
        emo_text = (
            f"(Stato interno mentre prendevo le decisioni: "
            f"curiosity={emotional_state.curiosity:.2f}, "
            f"confidence={emotional_state.confidence:.2f}, "
            f"fatigue={emotional_state.fatigue:.2f}, "
            f"frustration={emotional_state.frustration:.2f})"
        )
        parts.append(emo_text)

        if not any([analysis_plan, r_eda_result, modeling_result, r_churn_result, db_schema]):
            parts = [
                "Ho elaborato la tua richiesta ma non ho ancora risultati strutturati "
                "dai passi precedenti. Possiamo approfondire insieme specificando meglio il contesto."
            ]

        text = "\n".join(parts)

        output = {
            "user_visible_message": text,
            "stop_for_user_input": False,
        }

        delta = EmotionDelta(confidence=0.05, curiosity=0.02)
        return AgentResult(output_payload=output, emotion_delta=delta)
