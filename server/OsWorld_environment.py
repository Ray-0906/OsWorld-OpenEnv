# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Osworld Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import contextlib
import io
from typing import Any, Dict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OsworldAction, OsworldObservation, TaskLevel
except ImportError:
    from models import OsworldAction, OsworldObservation, TaskLevel


class OsworldEnvironment(Environment):
    """
    Data Cleaning Environment.
    Supports step-wise reward shaping, deterministic grading, and structured actions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Data Cleaning environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.files: Dict[str, str] = {}
        self.task_level: TaskLevel = TaskLevel.EASY
        self.screen_text = ""
        self.max_steps = 10

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def _setup_task(self):
        self.screen_text = f"Starting {self.task_level.value} data cleaning task."
        if self.task_level == TaskLevel.EASY:
            self.files = {"data.csv": "id,name\n1,alice\n1,alice\n2,bob\n"}
        elif self.task_level == TaskLevel.MEDIUM:
            self.files = {"data.csv": "id,val\n1,\n2,20\n3,\n"}
        else:
            self.files = {"data.csv": "id,val\n1,10\n1,10\n2,\n3,30\n"}

    def reset(self) -> OsworldObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Cycle tasks
        levels = [TaskLevel.EASY, TaskLevel.MEDIUM, TaskLevel.HARD]
        self.task_level = levels[self._reset_count % 3]
        self._setup_task()

        return OsworldObservation(
            screen_text=self.screen_text,
            files=self.files,
            current_task=f"Clean data.csv for {self.task_level.value} level.",  
            done=False,
            reward=0.0,
            score=self._get_score(),
        )

    def _get_score(self) -> float:
        import pandas as pd
        import io
        
        content = self.files.get("data.csv", "")
        # If the file is completely destroyed or unparseable, return 0.0
        try:
            df = pd.read_csv(io.StringIO(content))
        except Exception:
            return 0.0

        score = 0.0
        
        # 1. Define the Semantic "Golden" State
        if self.task_level == TaskLevel.EASY:
            expected_df = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        elif self.task_level == TaskLevel.MEDIUM:
            expected_df = pd.DataFrame({"id": [1, 2, 3], "val": [0, 20, 0]})
        else:
            expected_df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 0, 30]})

        try:
            # 2. Check Schema Integrity (Columns)
            if list(df.columns) == list(expected_df.columns):
                score += 0.2
            
            # 3. Handle Types to ensure fair comparison
            for col in expected_df.columns:
                if col in df.columns and expected_df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999).astype(int)

            # 4. Check Dataset Shape (Row Count)
            if len(df) == len(expected_df):
                score += 0.2

            # 5. Semantic Row Value Comparison 
            # Sort both so row order doesn't punish the agent if it rearranged them
            df_sorted = df.sort_values("id").reset_index(drop=True)
            exp_sorted = expected_df.sort_values("id").reset_index(drop=True)
            
            # Check how many exact rows match semantically, max out at 1.0 ratio
            # Use drop_duplicates on the merge so multiple 'Alice' records don't overinflate the join
            merged = df_sorted.drop_duplicates().merge(exp_sorted.drop_duplicates(), how='inner')
            matching_rows_ratio = min(1.0, len(merged) / len(exp_sorted) if len(exp_sorted) > 0 else 0)
            
            score += (0.6 * matching_rows_ratio)

            return min(1.0, round(score, 2))
        except Exception:
            return min(1.0, round(score, 2))

    def step(self, action: OsworldAction) -> OsworldObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = -0.05
        self.screen_text = f"Executed {action.action_type}."

        old_score = self._get_score()

        if action.action_type == "execute_python":
            code = action.payload.get("code", "")
            f = io.StringIO()
            try:
                # Need to explicitly import pandas inside the exec environment to support agent scripts natively 
                with contextlib.redirect_stdout(f):
                    local_vars = {"files": self.files}
                    exec("import pandas as pd\nimport io\nimport traceback\n" + code, {}, local_vars)
                self.screen_text = f.getvalue() or f"Successfully executed shape: {len(code)} chars."
            except Exception as e:
                import traceback
                self.screen_text = f"Python Execution Error:\n{traceback.format_exc()}"
                reward -= 0.2  # penalty for syntax error

        elif action.action_type == "remove_duplicates":
            filename = action.payload.get("filename", "data.csv")
            if filename in self.files:
                lines = self.files[filename].splitlines(keepends=True)
                seen = set()
                new_lines = []
                for line in lines:
                    if line not in seen:
                        seen.add(line)
                        new_lines.append(line)
                self.files[filename] = "".join(new_lines)

        elif action.action_type == "fill_nulls":
            filename = action.payload.get("filename", "data.csv")
            fill_val = action.payload.get("value", "0")
            if filename in self.files:
                self.files[filename] = self.files[filename].replace(",\n", f",{fill_val}\n")

        else:
            reward -= 0.2  # penalty for unknown action
            self.screen_text = "Unknown action."

        new_score = self._get_score()
        # The Shaping Function Φ(s) - scales by score jump or regression
        reward += (new_score - old_score)

        done = self._state.step_count >= self.max_steps or new_score >= 1.0     
        if done and new_score >= 1.0:
            reward += 10.0  # Massive terminal reward upon goal completion

        return OsworldObservation(
            screen_text=self.screen_text,
            files=self.files,
            current_task=f"Clean data.csv for {self.task_level.value} level.",
            done=done,
            reward=reward,
            score=new_score
        )
        return self._state