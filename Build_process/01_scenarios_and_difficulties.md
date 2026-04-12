# Episode Scenarios and Difficulty Modes

The Data Cleaning Environment presents AI agents with simulated programmatic challenges of varying complexity. There are **15 task variants** grouped into 3 difficulty tiers, procedurally generated with a fixed seed for reproducibility.

## Scenario Tiers

### 1. Easy Tier (4 variants)
Fundamental schema and targeted parsing corrections.

#### Variant A: Duplicate Removal
- **Goal:** Standardize columns (`id`, `name`) and remove duplicate rows.
- **Optimal Steps:** 2

#### Variant B: Format Normalization
- **Goal:** Strip whitespace and normalize strings to lowercase.
- **Optimal Steps:** 2

#### Variant C: Type Coercion
- **Goal:** Convert semantic strings (e.g., `"X yrs"` → int, `"Yes/No"` → bool).
- **Optimal Steps:** 3

#### Variant D: Column Rename
- **Goal:** Pure schema alignment, no data changes.
- **Optimal Steps:** 2

---

### 2. Medium Tier (7 variants)
Tasks combining structural issues with distinct semantic data quality problems.

#### Variant A: Missing Value Imputation
- **Goal:** Drop junk columns and fill nulls with `0`.
- **Optimal Steps:** 3

#### Variant B: Schema Repair
- **Goal:** Rename messy columns and strip extraneous ones.
- **Optimal Steps:** 3

#### Variant C: Constraint Enforcement
- **Goal:** Deduplicate records and clip values to valid ranges like `[0, 100]`.
- **Optimal Steps:** 4

#### Variant D: Multi-File Join
- **Goal:** Clean schema, perform an inner join across two CSVs.
- **Optimal Steps:** 5

#### Variant E: JSON Normalization
- **Goal:** Flatten deeply nested JSON into a clean tabular CSV.
- **Optimal Steps:** 4

#### Variant F: SQL Extraction
- **Goal:** Parse SQL dump, join tables in SQLite, and export as CSV.
- **Optimal Steps:** 4

#### Variant G: HTML Scraping
- **Goal:** Parse HTML table, clean whitespace, and export as CSV.
- **Optimal Steps:** 4

---

### 3. Hard Tier (4 variants)
Multi-step, multi-file dependency and destructive data recovery scenarios.

#### Variant A: Pipeline Recovery
- **Goal:** Fix columns + dedup + string cleaning + clipping + null fill simultaneously.
- **Optimal Steps:** 6

#### Variant B: Adversarial Corruption
- **Goal:** Syntactically valid but semantically impossible constraints — clip only.
- **Optimal Steps:** 5

#### Variant C: Cascading Pipeline
- **Goal:** String clean → null fill → currency conversion across two files.
- **Optimal Steps:** 6

#### Variant D: Log Parsing
- **Goal:** Extract structured records from unstructured system logs, filter invalid entries.
- **Optimal Steps:** 5

## Agent Constraints
- The agent only sees the files in the current workspace.
- The task objective is provided via natural language (`task_description`).
- Available libraries in the sandboxed Python execution: `pandas`, `json`, `io`, `sqlite3`, `re`, `bs4`.
- The agent primarily uses `execute_python` configuration to manipulate data, with optional read/inspect actions.
