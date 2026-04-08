import os
import json
from dotenv import load_dotenv
import openai
from client import OsworldEnv
from models import OsworldAction
from pydantic import BaseModel
from typing import Dict, Any

class Payload(BaseModel):
    # Defining a specific schema instead of Dict[str, Any] which breaks strict structured parsing
    filename: str | None = None
    n: int | None = None
    column: str | None = None
    value: str | None = None
    code: str | None = None

def sanitize_payload(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip trailing JSON syntax or markdown junk from string fields."""
    cleaned = {}
    for k, v in payload_dict.items():
        if isinstance(v, str) and v:
            # If it's the code field, firmly strip markdown fences
            if k == "code":
                v = v.strip()
                if v.startswith("```python"):
                    v = v[len("```python"):].strip()
                elif v.startswith("```"):
                    v = v[len("```"):].strip()
                if v.endswith("```"):
                    v = v[:-3].strip()
            # For purely string filenames, strip hallucinated JSON closures
            elif k in ["filename", "column", "value"]:
                v = v.split('}')[0].split(']')[0].strip().strip('"`')
        cleaned[k] = v
    return cleaned

class LLMAction(BaseModel):
    action_type: str
    payload: Payload

load_dotenv()

# The environment auto-cycles through 12 task variants across 3 difficulty tiers:
# (Easy: 4 tasks, Medium: 5 tasks, Hard: 3 tasks)
# It alternates tiers each reset (Easy -> Medium -> Hard -> Easy...)
# To ensure the baseline sees every single Medium variant (the max at 5),
# we need to run 5 * 3 = 15 episodes. (Easy and Hard tasks will naturally repeat).
NUM_EPISODES = 15

def main():
    # Initialize the OpenAI client to point to OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in environment.")
        return

    try:
        openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    except openai.OpenAIError as e:
        print(f"Client initialization error: {e}")
        return

    # Connect to the local environment server globally, so reset_count increments
    env = OsworldEnv(base_url="http://localhost:8000").sync()

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n==========================================")
        print(f"Episode {episode} / {NUM_EPISODES}")
        print(f"==========================================")
        
        # Reset cycles automatically through variants
        result = env.reset()
        obs = result.observation
        done = result.done

        # Print what task we got
        print(f"Task: {obs.current_task}")
        print(f"Initial Score: {obs.score:.4f}")

        step = 0
        history = []
        
        while not done:
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
            prompt = f"""
You are a specialized Data Engineering Agent. You must resolve data quality issues programmatically.
Here is your current observation:
{json.dumps(obs_dict, indent=2)}

Your goal is to transform the provided datasets to match a rigorous clean structure. You are evaluated on:
- Content Accuracy: Ensuring precision and recall of the data records.
- Schema Integrity: Validating column names, types, and strict ordering.
- Data Validity: Eliminating nulls, standardizing formats, and correcting types.
- Constraint Satisfaction: Enforcing uniqueness and logical range boundaries.

ENVIRONMENT REWARDS & REASONING:
- This is a Reinforcement Learning environment. You receive positive rewards for incremental progress toward the clean state.
- Efficiency is critical. Your final evaluation score is scaled by how few actions you take to reach the goal.
- Catastrophic data loss (e.g., dropping the majority of rows accidentally) results in a heavy negative penalty.
- Professional methodology is incentivized. It is highly recommended to inspect the data and schema before committing to permanent changes.

Available Action Types:
1. "inspect_schema": Check column names and types. Use "filename" in payload.
2. "view_head": Look at the first N rows. Use "filename" and "n" (default 5) in payload.
3. "read_file": Read the entire file content. Use "filename" in payload.
4. "preview_changes": Test your "code" without saving changes. (Zero risk, high transparency).
5. "execute_python": Perform permanent file mutations via Python code.
6. "remove_duplicates": Utility to deduplicate a file. Use "filename".
7. "fill_nulls": Utility to fill missing values. Use "filename" and "value".

PYTHON EXECUTION RULES:
For `execute_python` and `preview_changes`, your code runs in a sandboxed `exec()` environment.
- You have access to a `files` dictionary containing the file strings.
- You MUST read and write the CSV via this dict: `files["data.csv"]`
- Example Pattern:
  ```python
  import pandas as pd
  import io
  df = pd.read_csv(io.StringIO(files['data.csv']))
  # apply fixes...
  files['data.csv'] = df.to_csv(index=False)
  ```

Decide on the next action. Your response must be a single, valid JSON object.
"""
            user_msg = {"role": "user", "content": prompt}
            messages = [{"role": "system", "content": "You are a professional data cleaning engineer."}]
            messages.extend(history)
            messages.append(user_msg)

            try:
                response = openai_client.beta.chat.completions.parse(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    response_format=LLMAction,
                )
                
                # Update history so the agent remembers its actions
                history.append(user_msg)
                history.append({"role": "assistant", "content": response.choices[0].message.content})

                llm_action = response.choices[0].message.parsed
                payload_dict = llm_action.payload.model_dump(exclude_none=True)
                # Sanitize to prevent "data.csv}}]}" hallucinations
                payload_dict = sanitize_payload(payload_dict)
                
                action = OsworldAction(
                    action_type=llm_action.action_type,
                    payload=payload_dict
                )
            except Exception as e:
                print(f"Failed to query model: {e}")
                action = OsworldAction(
                    action_type="pass",
                    payload={}
                )
                
            step_result = env.step(action)
            obs = step_result.observation
            done = step_result.done
            reward = step_result.reward
            step += 1
            
            print(f"Step {step} | Action: {action.action_type} | Reward: {reward:+.4f} | Score: {obs.score:.4f}")
            
            if done:
                final_score = getattr(obs, "score", 0.0)
                print(f"\n--- Episode {episode} Finished ---")
                print(f"Final Score: {final_score:.4f}")

    # Cleanly close the environment client after all episodes
    if hasattr(env, "close"):
        env.close()

if __name__ == "__main__":
    main()