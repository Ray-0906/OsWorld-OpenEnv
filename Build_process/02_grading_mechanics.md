# Semantic Graders and Validating "Clean" Data

Unlike traditional code-evaluation systems that rely on brittle binary grading (e.g., exact byte-for-byte text matching where a single trailing whitespace fails the agent), this environment implements an autonomous **Semantic Grader**.

## The Problem With Exact String Matching
If the environment expected:
```csv
id,name
1,alice
2,bob
```
But the agent outputted:
```csv
id,name
1.0,alice
2.0,bob 
```
Or added differing index behavior, exact boolean checks would fail, causing the agent to receive a score of `0.0`. This stifles reinforcement learning optimization since the agent *did* accomplish the task semantically but failed a formatting constraint.

## The Semantic Grader Architecture
The score calculation engine operates on structural data properties rather than text. 

1. **Extraction**: The system reads `files['data.csv']` via `io.StringIO()` into a pandas DataFrame dynamically.
2. **Ground Truth Loading**: The environment instantiates a ground truth "expected" DataFrame alongside it based on the `current_task` state.
3. **Inner Merge Evaluation**: 
   The grader performs an automated inner join/merge between the Agent's dataset and the Ground Truth dataset.
   ```python
   # Conceptual evaluation
   merged = pd.merge(agent_df, expected_df, how='inner')
   ```
4. **Fractional Scoring**:
   The number of matching rows defines the proportional success. 
   - If the agent cleans half the data before making a mistake, they earn a proportional score (e.g., $0.5$).
   - **Exploit Prevention**: Duplicates are forcefully dropped in the evaluation layer (`drop_duplicates()`) to prevent the agent from artificially over-inflating their score by joining infinite matching rows.
5. **Score Publishing**: The internal score (from `0.0` to `1.0`) is passed firmly through the Pydantic type models into the OpenEnv `OsworldObservation.score` field, guaranteeing safe telemetric transmission to the client without dictionary-loss.