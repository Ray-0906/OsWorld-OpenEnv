# OsWorld Data Cleaning Environment

This is the **Data Cleaning Environment** built on the OpenEnv framework. It presents programmatic challenges of varying complexity where AI agents must read, diagnose, and clean data artifacts (`data.csv` files). 

The environment uses structured Pydantic inputs, sandboxed Python code execution, fractional reinforcement learning reward shaping, and an autonomous Semantic Grader to evaluate agents.

## Quick Start & Setup

The project is built using Python `uv` for lightning-fast dependency management.

1. **Install uv** (if you haven't already):
   ```bash
   pip install uv
   ```

2. **Clone and Install Dependencies**:
   ```bash
   # From the project root
   uv sync
   ```

## Environment Variables (.env)

The `baseline.py` script relies on an LLM to interact with the environment. It defaults to using OpenRouter (specifically the `gpt-4o-mini` model). 

You must create a `.env` file in the root of the project with your API key:

1. Create a file named `.env`:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   
   # If you adjust baseline.py to use standard OpenAI:
   # OPENAI_API_KEY=your_openai_api_key_here
   ```

*(Note: The environment server itself does not require an API key to run, only the baseline agent script requires it).*

## Running the Project

You can run the environment in two modular ways: dynamically instantiated by the client, or as a standalone WebSocket/HTTP FastAPI server.

### 1. Run the Baseline Agent (End-to-End)
This executes the LLM agent against the Easy, Medium, and Hard difficulty tracks. It spins up the server dynamically.
```bash
uv run baseline.py
```

### 2. Standalone Server Mode
If you are developing your own agent, or want to attach a separate script, run the server separately:
```bash
uvicorn server.app:app --port 8000 --reload
```

And in your agent script, connect to it:
```python
from client import OsworldEnv
# Connects to standard local uvicorn host
env = OsworldEnv(url="ws://localhost:8000")
env.reset()
```

## Documentation

For deep dives into how the underlying architecture works, please see our `Build_process/` documentation:

- [**Scenarios and Difficulties**](Build_process/01_scenarios_and_difficulties.md): Details on file states and expected outputs for Easy, Medium, and Hard tracks.
- [**Semantic Grading Mechanics**](Build_process/02_grading_mechanics.md): How the Pandas inner-join grader eliminates the vulnerabilities of string-matching. 
- [**Reward Shaping & Scoring**](Build_process/03_reward_shaping.md): How mathematical potential-bounds and execution penalties enforce optimal reasoning paths.

## Project Structure
```text
OsWorld/
├── Build_process/         # Architectural documentation
├── __init__.py            # Module exports
├── client.py              # OsworldEnv client
├── models.py              # Strict Action and Observation Pydantic models
├── baseline.py            # Reference agent using OpenRouter LLM
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Requirements
└── server/
    ├── OsWorld_environment.py  # Core environment logic and Graders
    └── app.py             # FastAPI App
```