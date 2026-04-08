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

> A benchmark environment for training and evaluating LLM agents on multi-step data engineering and cleaning tasks.

---

## 🧠 Architecture Overview

The **OsWorld Data Cleaning Environment** models data cleaning as a Markov Decision Process (MDP).

- **State Representation**: Workspace files (CSV, JSON, SQL, HTML, logs) and task description  
- **Action Space**: Structured actions (inspection, Python execution, utilities)  
- **Semantic Grading**: Multi-component Φ (Phi) scoring system  
- **Reward System**: Delta-based shaping with penalties and efficiency-scaled terminal rewards  

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

## 📋 Task Scenarios

The environment contains **15 task variants** across 3 difficulty tiers.

Each `reset()` returns a deterministic task instance, cycling through predefined variants.

> During benchmarking (`inference.py`), only a **single episode** is executed per run to ensure strict evaluation consistency.

---

### 🌟 Easy Tier (4 Variants)

* **Duplicate Removal**
  Standardize columns (`id`, `name`) and remove duplicates

* **Format Normalization**
  Strip whitespace and normalize strings to lowercase

* **Type Coercion**
  Convert semantic strings into correct types

* **Column Rename**
  Pure schema alignment

---

### ⚡ Medium Tier (7 Variants)

* Missing value imputation
* Schema repair
* Constraint enforcement
* Multi-file joins
* JSON normalization
* SQL extraction
* HTML scraping

---

### 🔥 Hard Tier (4 Variants)

* Corrupted pipeline recovery
* Adversarial data fixing
* Cascading multi-file transformations
* Log parsing

---

## 🔁 Interaction Loop

Each episode follows:

1. Agent receives observation (files + task)
2. Agent emits an action
3. Environment executes action in sandbox
4. Returns updated state, reward, and score

Loop continues until termination or max steps.

---

## 📊 Evaluation System

### Φ Score (Semantic Grading)

$$
\Phi = 0.4 \cdot content + 0.2 \cdot schema + 0.2 \cdot validity + 0.2 \cdot constraints - penalty
$$

* All components normalized to **[0, 1]**
* Final Φ is clamped to **[0, 1]**
* **Task is solved when Φ = 1.0**

---

### Component Breakdown

**Content (40%)**
F1-based row matching (precision + recall)

**Schema (20%)**
Column overlap + ordering consistency (capped ≤ 1.0)

**Validity (20%)**

* Null handling
* Type correctness
* Format consistency

**Constraints (20%)**

* Uniqueness
* Value ranges
* Task-specific rules

---

### Anti-Cheat Penalty

$$
penalty = \min\left(0.3,\; 0.1 \cdot \frac{\max(0, n_{agent} - n_{expected})}{n_{expected}}\right)
$$

Prevents:

* Row inflation
* Duplicate exploitation
* Partial-output hacks

---

## 🎯 Reward Function

The reward at each step is:

```
R = step_penalty
  + (new_score - old_score)
  + regression_penalty (if score drops)
  + error_penalty
  + destructive_penalty
  + terminal_bonus (if done)
```

---

### 🔥 Terminal Reward Scaling

When the episode ends, a **terminal bonus** is applied:

```
terminal_bonus = α * final_score * efficiency_factor
```

Where:

* **final_score = Φ (final semantic score)**
* **α = terminal scaling constant (e.g., 5.0)**

---

### ⚡ Efficiency Factor

Encourages fewer steps:

```
efficiency_factor = max(0.3, 1 - steps_used / max_steps)
```

Properties:

* Faster solutions → higher reward
* Slow agents are penalized but not zeroed
* Minimum floor prevents reward collapse

---

### 🧠 Reward Properties

* **Dense feedback** → incremental progress rewarded
* **Regression penalty** → discourages breaking correct states
* **Execution penalties** → punishes unsafe/destructive actions
* **Efficiency scaling** → promotes optimal planning

---

## 🤖 Agent Constraints

* Agent only sees provided files
* Tasks are defined in `current_task`
* Execution occurs in sandboxed Python
* Available libraries: `pandas`, `io`, etc.
* Agent primarily uses `execute_python`, with optional inspection steps

---

## 🚀 Usage

### Benchmark Mode

```bash
uv run inference.py
```

Outputs strict logs:

```
[START]
[STEP]
[END]
```

---

### Evaluation

```bash
uv run python eval.py
```

---

### Run Server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 📂 Project Structure

```
OsWorld/
├── server/
│   ├── tasks.py
│   ├── graders.py
│   └── OsWorld_environment.py
├── models.py
├── client.py
├── inference.py
└── openenv.yaml
```

---

## 🧠 Design Philosophy

* Semantic correctness over exact matching
* Robustness against reward hacking
* Real-world data complexity
* Multi-step reasoning evaluation
* Efficiency-aware agent behavior

---

*Built for OpenEnv Hackathon | Designed for serious agent evaluation*