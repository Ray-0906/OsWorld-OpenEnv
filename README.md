---
title: OsWorld-OpenEnv
emoji: 🌍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# OsWorld Data Cleaning Environment

> A benchmark environment for training and evaluating LLM agents on multi-step, real-world data engineering tasks.

---

## 🧠 Why This Environment?

Data cleaning is one of the most frequent, high-stakes tasks a human data engineer performs — and one of the hardest for LLM agents to do correctly. Unlike toy benchmarks, this environment requires agents to:

- Reason about **data quality** across multiple formats (CSV, JSON, SQL, HTML, logs)
- Plan **multi-step pipelines** where earlier mistakes compound
- Navigate **adversarial corruption** that is structurally valid but semantically broken
- Operate efficiently — not just correctly

The environment is designed to produce a **smooth, dense reward signal** that reflects genuine incremental progress, making it suitable for GRPO and other RL training regimes.

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install uv
```

### 2. Installation
```bash
uv sync
```

### 3. Environment Configuration

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
API_BASE_URL=https://router.huggingface.co/v1

# Optional fallback
OPENROUTER_API_KEY=your_key
```

---

## 📂 Project Structure

```
OsWorld/
├── server/
│   ├── tasks.py                  # 15 procedurally generated task variants
│   ├── graders.py                # SemanticGrader — multi-component Phi scoring
│   ├── reward.py                 # RewardCalculator — dense reward shaping
│   └── OsWorld_environment.py   # Core MDP environment (step, reset, state)
├── models.py                     # Typed Pydantic models (Observation, Action, Reward)
├── client.py                     # OpenEnv client
├── inference.py                  # Baseline inference script
├── openenv.yaml                  # OpenEnv spec metadata
└── Dockerfile
```

---

## 🔁 Interaction Loop

The environment models data cleaning as a **Markov Decision Process (MDP)**:

```
reset() → initial observation
  └── agent receives: files (workspace) + task description
        └── agent emits: action
              └── step(action) → (observation, reward, done, info)
                    └── repeat until done or max_steps reached
```

- `reset()` → returns initial observation, loads task files into workspace
- `step(action)` → executes action, scores result, returns reward
- `state()` → returns current workspace state

---

## 👁️ Observation Space

Each observation is a typed Pydantic model containing:

| Field | Type | Description |
|---|---|---|
| `files` | `Dict[str, str]` | Current workspace file contents keyed by filename |
| `task_description` | `str` | Natural language description of the cleaning objective |
| `step_count` | `int` | Steps taken so far in the episode |
| `current_score` | `float` | Current Phi score [0, 1] |
| `done` | `bool` | Whether the episode has terminated |

---

## ⚡ Action Space

Actions are structured Pydantic models. The agent may emit:

| Action | Description |
|---|---|
| `execute_python` | Execute a Python snippet in the sandboxed environment |
| `preview_changes` | Test code without saving changes (zero-risk dry run) |
| `inspect_schema` | View column names and dtypes of a file |
| `view_head` | Preview the first N rows of a file |
| `read_file` | Read raw file contents |
| `remove_duplicates` | Utility to deduplicate a file by exact row match |
| `fill_nulls` | Utility to fill missing values in a file |

The agent primarily uses `execute_python` to manipulate files. Inspection actions are available and rewarded when used strategically on the first step.

---

## 📋 Task Scenarios

The environment contains **15 task variants** across 3 difficulty tiers, procedurally generated with a fixed seed for reproducibility.

Each `reset()` returns a deterministic task instance, cycling through variants.

> During benchmarking (`inference.py`), a **single episode** is executed per run to ensure strict evaluation consistency.

---

### 🌟 Easy Tier (4 Variants)

| Task | Description | Optimal Steps |
|---|---|---|
| Duplicate Removal | Standardize columns (`id`, `name`), remove duplicate rows | 2 |
| Format Normalization | Strip whitespace, normalize strings to lowercase | 2 |
| Type Coercion | Convert `"X yrs"` → int, `"Yes/No"` → bool | 3 |
| Column Rename | Pure schema alignment, no data changes | 2 |

---

### ⚡ Medium Tier (7 Variants)

| Task | Description | Optimal Steps |
|---|---|---|
| Missing Value Imputation | Drop junk columns, fill nulls with 0 | 3 |
| Schema Repair | Rename messy columns, strip extraneous ones | 3 |
| Constraint Enforcement | Deduplicate, clip values to [0, 100] | 4 |
| Multi-File Join | Clean schema, inner join across two CSVs | 5 |
| JSON Normalization | Flatten deeply nested JSON to tabular CSV | 4 |
| SQL Extraction | Parse SQL dump, join tables in SQLite, export CSV | 4 |
| HTML Scraping | Parse HTML table, clean whitespace, export CSV | 4 |

---

### 🔥 Hard Tier (4 Variants)

| Task | Description | Optimal Steps |
|---|---|---|
| Pipeline Recovery | Fix columns + dedup + string cleaning + clipping + null fill simultaneously | 6 |
| Adversarial Corruption | Structurally valid but semantically impossible values — clip only | 5 |
| Cascading Pipeline | String clean → null fill → currency conversion across two files | 6 |
| Log Parsing | Extract structured records from unstructured logs, filter invalid entries | 5 |

---

## 📊 Evaluation System — Φ (Phi) Score

The SemanticGrader computes a multi-component score designed to give **smooth partial credit** at every stage of cleaning.

### Formula

```
structural = 0.35 * schema_score + 0.30 * validity_score + 0.35 * constraint_score

gated_content = content_score * (0.6 + 0.4 * schema_score)

base = (0.55 * gated_content + 0.45 * structural) * row_balance

Phi = base ** 1.05 - extra_row_penalty
```

Phi is clamped to **[0.0001, 0.9999]**. Task is considered solved at **Phi ≥ 0.99**.

---

### Component Breakdown

**Content Score (55% weight via gated blend)**

F1-based multiset row matching using loose column normalization. Handles case/whitespace/underscore variants so `"ID"` and `"id"` are treated as the same column. Coverage factor penalizes missing columns naturally.

**Schema Score**

Blended strict + loose Jaccard similarity (75% strict, 25% loose) over column names. Order bonus (+0.2) awarded when column order matches exactly.

**Validity Score (W = 0.30)**

- Null rate in required columns
- Numeric type correctness
- String formatting (stripped and lowercased)

**Constraint Score (W = 0.35)**

- Uniqueness of key columns
- Value range enforcement
- Required column presence

**Row Balance Multiplier**

```
row_balance = max(0.0, 1.0 - |n_agent - n_expected| / max(n_agent, n_expected))
```

Applied as a **multiplier** on the full score — row count mismatches penalize all components simultaneously.

**Anti-Cheat Penalty**

```
penalty = min(0.35, 0.14 * max(0, n_agent - n_expected) / n_expected)
```

Prevents row inflation, duplicate exploitation, and partial-output hacks.

---

## 🎯 Reward Function

The reward at each step is **dense** — the agent receives meaningful signal on every action, not just at episode end.

```
R = step_penalty                          # -0.03 always (efficiency pressure)
  + delta                                 # score improvement (new_score - old_score)
  + regression_penalty (if delta < 0)    # -0.1 for breaking correct state
  + error_penalty (if error/unknown)      # -0.2
  + destructive_penalty (if destructive) # -0.5
  + inspect_first_bonus (step 1 only)    # +0.05 for inspecting data first
  + terminal_bonus (if solved)           # efficiency-scaled (see below)
```

---

### Terminal Bonus

Only fires when `Phi ≥ 0.99` (task solved):

```
efficiency_ratio = min(1.0, optimal_steps / actual_steps)
efficiency_ratio = max(0.2, efficiency_ratio)   # floor prevents zero reward for slow but correct agents
terminal_bonus = terminal_reward * efficiency_ratio   # default terminal_reward = 2.0
```

| Scenario | Efficiency Ratio | Terminal Bonus |
|---|---|---|
| Solved in optimal steps | 1.0 | 2.0 |
| Solved in 2x optimal steps | 0.5 | 1.0 |
| Solved very slowly | 0.2 (floor) | 0.4 |
| Not solved | — | 0 |

---

### Reward Properties

- **Dense feedback** — incremental Phi improvements are rewarded every step
- **Regression penalty** — discourages breaking previously correct states
- **Execution penalties** — punishes unsafe, destructive, or unknown actions
- **Efficiency scaling** — promotes optimal planning without punishing successful slow agents to zero

---

## 🤖 Agent Constraints

- Agent only sees files in the current workspace
- Task objective is provided in natural language via `task_description`
- Execution occurs in a sandboxed Python environment
- Available libraries: `pandas`, `json`, `io`, `sqlite3`, `re`, `bs4`
- Agent primarily uses `execute_python`, with optional inspection steps

---

## 🚀 Usage

### Benchmark Mode

```bash
uv run inference.py
```

Produces strict per-step logs:

```
[START]
[STEP]
[END]
```

### Evaluation

```bash
uv run python eval.py
```

### Run Server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 📈 Baseline Scores

Baseline scores on a single episode per task variant using `Qwen/Qwen2.5-72B-Instruct`:

| Tier | Mean Phi | Mean Steps |
|---|---|---|
| Easy | 0.9999 | 2 |
| Medium | 0.9999 | 8 |
| Hard | 0.3965 | 10 |

> Run `inference.py` to populate baseline scores for your model.

---

## 🧠 Design Philosophy

- **Semantic correctness over exact matching** — partial credit at every component
- **Robustness against reward hacking** — row balance multiplier + anti-cheat penalty
- **Real-world data complexity** — multi-format, multi-file, adversarial inputs
- **Multi-step reasoning evaluation** — cascading dependencies reward planning
- **Efficiency-aware agent behavior** — terminal bonus scales with solution quality

---

*Built for OpenEnv Hackathon | Designed for serious agent evaluation*