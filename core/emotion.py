from __future__ import annotations

from .models import EmotionalState, AgentRun, EmotionDelta, AgentRunStatus


class EmotionalEngine:
    """
    Motore emotivo non parametrico.
    Regole hard-coded ispirate ad Aurion: successi/fallimenti, tipo di agente,
    tipo di interazione.
    """

    def apply_decay_between_turns(self, state: EmotionalState) -> EmotionalState:
        """
        Decadimento leggero tra turni: scarica fatica/frustrazione,
        riporta mood verso neutro, energia verso un baseline.
        """
        # scarica fatica/frustrazione
        state.fatigue *= 0.9
        state.frustration *= 0.9

        # mood torna piano verso 0 (neutro)
        state.mood *= 0.95

        # energia torna verso baseline ~0.6
        baseline_energy = 0.6
        state.energy += (baseline_energy - state.energy) * 0.1

        # drive si normalizzano un po'
        state.social_need *= 0.98
        state.playfulness *= 0.98
        state.learning_drive = min(1.0, state.learning_drive * 0.99 + 0.01)

        return state

    def update_on_agent_run(
        self,
        state: EmotionalState,
        run: AgentRun,
    ) -> EmotionalState:
        delta = EmotionDelta()

        is_success = run.status == AgentRunStatus.SUCCESS
        agent_name = run.agent_name.lower()

        if is_success:
            # successi fanno bene al mood/energia/confidenza
            delta.confidence += 0.05
            delta.curiosity  += 0.02
            delta.fatigue    += 0.005
            delta.frustration -= 0.02

            delta.mood   += 0.05
            delta.energy += 0.03

            # imparare qualcosa → learning_drive
            delta.learning_drive += 0.02
        else:
            # fallimenti logorano
            delta.confidence  -= 0.05
            delta.frustration += 0.08
            delta.fatigue     += 0.03

            delta.mood   -= 0.08
            delta.energy -= 0.02

            # aumenta bisogno di conforto / relazione
            delta.social_need += 0.05

        # modulazioni per tipo di agent (hard-coded)
        if "requirements" in agent_name:
            if not is_success:
                delta.frustration += 0.05
                delta.mood        -= 0.03
        if "analysis_planner" in agent_name and is_success:
            delta.curiosity      += 0.03
            delta.learning_drive += 0.03
        if "chat_agent" in agent_name:
            # parlare con l'utente scarica un po' social_need
            delta.social_need -= 0.02
            # se la chat è leggera (lo vedremo da contenuto, in futuro),
            # qui potresti anche aumentare playfulness.

        state.apply_delta(delta)
        return state
