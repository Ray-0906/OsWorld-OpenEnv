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
    column: str | None = None
    value: str | None = None
    code: str | None = None

class LLMAction(BaseModel):
    action_type: str
    payload: Payload

load_dotenv()

def main():
    # Initialize the OpenAI client to point to OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in environment.")
        return

    try:
        # We use the standard openai client, but redirect it to OpenRouter
        openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    except openai.OpenAIError as e:
        print(f"Client initialization error: {e}")
        return

    difficulties = ["easy", "medium", "hard"]

    for difficulty in difficulties:
        print(f"\n==========================================")
        print(f"Running difficulty: {difficulty.upper()}")
        print(f"==========================================")
        
        # Connect to the local environment server using the sync wrapper
        env = OsworldEnv(base_url="http://localhost:8000").sync()

        # Reset with specific difficulty option so the backend picks Easy/Medium/Hard
        result = env.reset(options={"difficulty": difficulty})
        obs = result.observation
        done = result.done

        step = 0
        
        while not done:
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
            
            prompt = f"""
You are an expert data cleaning bot. Here is your current observation:
{json.dumps(obs_dict, indent=2)}

You must solve the current_task by writing Python code to fix the dataset.
Your goal is to transform the CSV semantically to match clean structures exactly. 
If the file relies on `val` but the current task only involves `id,name`, do not hallucinate a `val` column.

Available actions include:
- action_type: "execute_python", payload: {{"code": '''import pandas as pd\\nimport io\\ndf = pd.read_csv(io.StringIO(files["data.csv"]))\\n# Clean dataframe...\\nfiles["data.csv"] = df.to_csv(index=False)'''}}
- action_type: "remove_duplicates", payload: {{"column": "id"}}
- action_type: "fill_nulls", payload: {{"column": "val", "value": "0"}}

Decide on the next action to take to progress data cleaning. Provide valid details.
"""
            
            # Using structured output parsing directly into OsworldAction equivalents
            try:
                response = openai_client.beta.chat.completions.parse(
                    # Make sure to use the OpenRouter model prefix!
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional data cleaning engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=LLMAction,
                )
                llm_action = response.choices[0].message.parsed
                
                # Convert the Pydantic Payload model to a dict, removing completely empty None values
                payload_dict = llm_action.payload.model_dump(exclude_none=True)
                
                action = OsworldAction(
                    action_type=llm_action.action_type,
                    payload=payload_dict
                )
            except Exception as e:
                print(f"Failed to query model: {e}")
                # Fallback action
                action = OsworldAction(
                    action_type="pass",
                    payload={}
                )
                
            # print(f"Payload: {action.payload}")
            step_result = env.step(action)
            obs = step_result.observation
            done = step_result.done
            reward = step_result.reward
            step += 1
            
            print(f"Step {step} | Task: {difficulty} | Action: {action.action_type} | Reward: {reward:.4f}")
            
            if done:
                # Retrieve the native explicit score float
                final_score = getattr(obs, "score", 0.0)
                print(f"\n--- Episode Finished ({difficulty}) ---")
                print(f"Final Score (0.0 to 1.0): {final_score:.4f}")

        # Cleanly close the environment client before next difficulty
        if hasattr(env, "close"):
            env.close()

if __name__ == "__main__":
    main()
