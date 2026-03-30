# Episode Scenarios and Difficulty Modes

The Data Cleaning Environment presents AI agents with simulated programmatic challenges of varying complexity. The agent is exposed to `data.csv` files via the environment state and must issue Python execution commands to clean the data.

## Task Difficulties

There are three primary difficulty tracks, each mimicking real-world data noise:

### 1. Easy Mode
- **Scenario:** The target CSV contains exact row duplicates. 
- **Goal:** The agent must safely remove duplicate records without disrupting the core data schema.
- **Example Start State:** 
  ```csv
  id,name
  1,alice
  1,alice
  2,bob
  ```
- **Expected End State:** Unique records for each entity.

### 2. Medium Mode
- **Scenario:** The target CSV is plagued by missing values (nulls/NaN) inside primary data columns.
- **Goal:** The agent must implement an imputation strategy (e.g., filling nulls with zeroes or a designated baseline value).
- **Example Start State:**
  ```csv
  id,val
  1,
  2,20
  3,
  ```
- **Expected End State:** All blank cells cleanly imputed, ensuring downstream data pipelines wouldn't crash.

### 3. Hard Mode
- **Scenario:** The dataset is fully compromised, exhibiting both structural duplication and pervasive missing values.
- **Goal:** The agent must execute a multi-step semantic cleaning strategy, often chaining `drop_duplicates` and `fillna` logic sequentially.
- **Example Start State:** A large table mixing Easy and Medium impurities.
- **Expected End State:** A mathematically sound representation matching a pre-computed perfect DataFrame.

## Agent Constraints
- The agent only "sees" the files exposed in the `files` observation dict.
- The agent must use strict `action_types` (like `execute_python`) formatted correctly.
- All actions execute inside a dynamic local python scope injected with core libraries (`pandas`, etc.), creating a true programmatic sandbox.