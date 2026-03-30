# OsWorld Data Cleaning Environment Architecture

This `Build_process` directory contains details describing the environment's internal mechanics. This system represents an OpenEnv-compliant reinforcement learning and heuristic evaluation wrapper targeting automated programmatic tasks.

### Documentation Index
- [`01_scenarios_and_difficulties.md`](./01_scenarios_and_difficulties.md): Breakdowns of the active datasets, task types, and expected edge states the agent operates within.
- [`02_grading_mechanics.md`](./02_grading_mechanics.md): The internal engineering behind the Semantic DataFrame Grader overcoming the vulnerabilities of brute-force binary string validation.
- [`03_reward_shaping.md`](./03_reward_shaping.md): Deep-dive into the thermodynamics-based $\Delta \Phi$ continuous scoring matrices, terminal bonuses, and step-decay penalty structures.