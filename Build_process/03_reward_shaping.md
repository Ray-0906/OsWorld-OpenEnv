# Reward Functions and Reinforcement Mechanisms

The environment provides a **dense** reward signal at each step, making it ideal for reinforcement tracking (like GRPO training). The multi-component grader natively guides the reward distribution on each individual step instead of clustering it at episode completion.

## Core Formula

The step-by-step reward formula works out to:
```text
R = step_penalty                          # -0.03 always (efficiency pressure)
  + delta                                 # score improvement (new_score - old_score)
  + regression_penalty (if delta < 0)    # -0.1 for breaking correct state
  + error_penalty (if error/unknown)      # -0.2
  + destructive_penalty (if destructive) # -0.5
  + inspect_first_bonus (step 1 only)    # +0.05 for inspecting data first
  + terminal_bonus (if solved)           # efficiency-scaled
```

## Reward Components

### 1. Differential Rewards ($\Delta\Phi$)
The backbone of the reward structure is progress-bound differential. Incremental $\Phi$ improvements directly result in a scaled reward.

### 2. Step, Regression & Execution Penalties
- **Step Penalty (-0.03):** Ensures pressure for finding the fastest, optimal track.
- **Regression Penalty (-0.1):** Triggers if an action regresses or resets correct logic ($\Delta\Phi < 0$), heavily punishing random execution and file truncation.
- **Error Penalty (-0.2):** Standard assessment on syntax crashes or invalid execution.
- **Destructive Penalty (-0.5):** Assessed when the baseline data structure is actively wiped or corrupted.

### 3. Execution Bonuses
- **Inspect-First (+0.05):** Added on the first step if the agent elects to run `read_file`, `inspect_schema`, or `view_head` to comprehend constraints. Proactively reinforces planning behaviors.
- **Terminal Bonus:** Fires when $\Phi \geq 0.99$. 

The Terminal Bonus actively factors in the user's ratio against "optimal steps" with a forced minimum limit to still reward slow successful agents.
```text
efficiency_ratio = min(1.0, optimal_steps / actual_steps)
efficiency_ratio = max(0.2, efficiency_ratio)   # floor prevents zero reward for slow agents
terminal_bonus = terminal_reward * efficiency_ratio   # base terminal_reward = 2.0
```

| Scenario | Efficiency Ratio | Terminal Bonus |
|---|---|---|
| Solved in optimal steps | 1.0 | 2.0 |
| Solved in 2x optimal steps | 0.5 | 1.0 |
| Solved very slowly | 0.2 (floor) | 0.4 |
| Not solved | — | 0 |
