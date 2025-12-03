# agents/diagnostics_agent.py
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List
import statistics
import time

from core.agents_base import Agent, AgentResult
from core.models import (
    EmotionalState,
    EmotionDelta,
    ConversationContext,
    AgentRunStatus,
    MemoryType,
    MemoryScope,
)
from core.memory import MemoryEngine
from core.llm_provider import LLMProvider


class DiagnosticsAgent(Agent):
    """
    Agent diagnostico avanzato.
    Monitora:
      - fallimenti e inefficienze degli agent
      - degrado prestazionale
      - mismatch input/output
      - problemi di routing
      - stato emotivo anomalo
    Scrive alert nella memoria procedurale.
    """

    name = "diagnostics_agent"
    description = "Analizza i run, individua anomalie, inefficienze e problemi sistemici."

    # ============================================
    # MAIN EXECUTION
    # ============================================
    def _run_impl(
        self,
        input_payload: Dict[str, Any],
        context: ConversationContext,
        memory: MemoryEngine,
        llm: LLMProvider,
        emotional_state: EmotionalState,
    ) -> AgentResult:

        lookback = int(input_payload.get("lookback", 200))
        runs = memory.get_recent_agent_runs(limit=lookback)

        if not runs:
            return AgentResult(
                output_payload={
                    "user_visible_message": "Non sono disponibili abbastanza esecuzioni per eseguire una diagnostica evoluta.",
                    "diagnostics": {},
                },
                emotion_delta=EmotionDelta(confidence=-0.01),
            )

        # ======================================================
        # 1) STATISTICHE DI BASE
        # ======================================================
        by_agent, last_error = self._compute_failures(runs)
        perf_stats = self._compute_performance(runs)
        io_issues = self._compute_io_problems(runs)
        routing_issues = self._compute_routing_issues(context, runs)

        emotional_anomalies = self._compute_emotional_anomalies(emotional_state)

        # ======================================================
        # COSTRUZIONE MESSAGGIO PER L’UTENTE
        # ======================================================
        text_lines = ["📊 *Diagnostica avanzata del sistema*"]

        text_lines.append("\n🔧 **Agenti più problematici**")
        for item in by_agent[:5]:
            text_lines.append(
                f"- **{item['agent_name']}** → {item['failures']}/{item['total_runs']} "
                f"fallimenti ({item['failure_rate']:.1%})"
            )
            if item["last_error"]:
                text_lines.append(f"    ultimo errore: {item['last_error'][:120]}")

        text_lines.append("\n🐌 **Inefficienze / lentezza**")
        for slow in perf_stats["slow_agents"]:
            text_lines.append(
                f"- {slow['agent_name']} → avg {slow['avg_duration']:.2f}s "
                f"(global avg {perf_stats['global_avg']:.2f}s)"
            )

        if io_issues:
            text_lines.append("\n💬 **Problemi di comunicazione input/output**")
            for issue in io_issues:
                text_lines.append(f"- {issue}")

        if routing_issues:
            text_lines.append("\n🗺️ **Problemi di routing**")
            for issue in routing_issues:
                text_lines.append(f"- {issue}")

        if emotional_anomalies:
            text_lines.append("\n❤️ **Anomalie emotive**")
            for issue in emotional_anomalies:
                text_lines.append(f"- {issue}")

        diagnostic_report = "\n".join(text_lines)

        # ======================================================
        #  SCRITTURA DI ALERT IN MEMORIA PROCEDURALE
        # ======================================================
        memory.store_item(
            scope=MemoryScope.GLOBAL,
            type_=MemoryType.PROCEDURAL,
            key="diagnostic_alert",
            content=diagnostic_report,
            metadata={"severity": "warning"},
        )

        # Delta emotivo: diagnostica → aumenta curiosità, ma segnala fatica
        delta = EmotionDelta(
            curiosity=0.04,
            frustration=0.01,
            fatigue=0.01
        )

        return AgentResult(
            output_payload={
                "user_visible_message": diagnostic_report,
                "diagnostics": {
                    "failures": by_agent,
                    "performance": perf_stats,
                    "io_issues": io_issues,
                    "routing_issues": routing_issues,
                    "emotional_issues": emotional_anomalies,
                },
            },
            emotion_delta=delta,
        )

    # ======================================================
    # COMPONENTI DIAGNOSTICI
    # ======================================================

    def _compute_failures(self, runs):
        """Aggrega fallimenti e calcola tassi + ultimo errore."""
        by_agent = defaultdict(Counter)
        last_error = {}

        for r in runs:
            by_agent[r.agent_name][r.status.value] += 1
            if r.status == AgentRunStatus.FAILURE:
                last_error[r.agent_name] = r.output_payload.get("error", "")

        scored = []
        for agent_name, counter in by_agent.items():
            total = sum(counter.values())
            failures = counter.get("failure", 0)
            failure_rate = failures / total if total > 0 else 0.0
            scored.append(
                {
                    "agent_name": agent_name,
                    "total_runs": total,
                    "failures": failures,
                    "failure_rate": failure_rate,
                    "last_error": last_error.get(agent_name, ""),
                }
            )

        scored.sort(key=lambda x: x["failure_rate"], reverse=True)
        return scored, last_error

    def _compute_performance(self, runs):
        durations = []
        per_agent = defaultdict(list)

        for r in runs:
            try:
                dur = (r.finished_at - r.started_at).total_seconds()
                durations.append(dur)
                per_agent[r.agent_name].append(dur)
            except Exception:
                continue

        if not durations:
            return {"global_avg": 0, "slow_agents": []}

        global_avg = statistics.mean(durations)
        slow_agents = []

        for agent, ds in per_agent.items():
            if ds:
                avg = statistics.mean(ds)
                if avg > global_avg * 1.8:  # threshold: 80% più lento del normale
                    slow_agents.append({"agent_name": agent, "avg_duration": avg})

        return {"global_avg": global_avg, "slow_agents": slow_agents}

    def _compute_io_problems(self, runs):
        issues = []
        for r in runs:
            if not r.input_payload:
                issues.append(f"{r.agent_name}: input_payload vuoto")
            if not r.output_payload:
                issues.append(f"{r.agent_name}: output_payload vuoto o mancante")
        return issues

    def _compute_routing_issues(self, context: ConversationContext, runs):
        issues = []
        if context.plan is None:
            return issues

        # esempio semplice: agent falliti subito dopo essere stati selezionati dal Router
        recent_failures = [r for r in runs[-20:] if r.status == AgentRunStatus.FAILURE]
        for r in recent_failures:
            issues.append(f"{r.agent_name} ha fallito immediatamente dopo routing")

        return issues

    def _compute_emotional_anomalies(self, emo: EmotionalState):
        issues = []
        if emo.fatigue > 0.75:
            issues.append("fatigue molto alta → possibile degrado cognitivo")
        if emo.frustration > 0.7:
            issues.append("frustrazione elevata → molte pipeline problematiche")
        if emo.confidence < 0.2:
            issues.append("confidence troppo bassa → il sistema dubita di se stesso")
        if emo.curiosity > 0.85:
            issues.append("curiosity elevata → rischio di loop esplorativi")
        return issues