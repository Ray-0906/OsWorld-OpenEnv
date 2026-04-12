import io
import re
from collections import Counter
from typing import Dict, Any

import pandas as pd


class SemanticGrader:
    W_SCHEMA = 0.35
    W_VALIDITY = 0.30
    W_CONSTRAINT = 0.35

    def _raw_col(self, name: Any) -> str:
        s = str(name).strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    def _loose_col(self, name: Any) -> str:
        return self._raw_col(name).lower()

    def _canon_df(self, df: pd.DataFrame, loose: bool = False) -> pd.DataFrame:
        out = df.copy()
        if loose:
            out.columns = [self._loose_col(c) for c in out.columns]
        else:
            out.columns = [self._raw_col(c) for c in out.columns]
        return out

    def get_score(
        self,
        files: Dict[str, str],
        expected_df: pd.DataFrame,
        constraints: Dict[str, Any],
    ) -> float:
        target_file = constraints.get("target_file", "data.csv")
        content = files.get(target_file, "")

        try:
            df = pd.read_csv(io.StringIO(content))
        except Exception:
            return 0.0001

        if len(df) == 0 and len(expected_df) > 0:
            return 0.0001

        content_score = self._content_score(df, expected_df)
        schema_score = self._schema_score(df, expected_df, constraints)
        validity_score = self._validity_score(df, expected_df, constraints)
        constraint_score = self._constraint_score(df, constraints)
        extra_penalty = self._extra_row_penalty(df, expected_df)
        row_balance = self._row_balance(df, expected_df)

        structural = (
            self.W_SCHEMA * schema_score
            + self.W_VALIDITY * validity_score
            + self.W_CONSTRAINT * constraint_score
        )

        # Soft gate: schema matters, but cannot dominate.
        gated_content = content_score * (0.6 + 0.4 * schema_score)

        # Additive blend: avoids one component crushing the others.
        base = (0.55 * gated_content + 0.45 * structural) * row_balance

        # Mild compression only.
        phi = base ** 1.05

        phi -= extra_penalty

        return min(0.9999, max(0.0001, round(phi, 4)))

    def _row_balance(self, df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        if len(df) == 0 and len(expected_df) == 0:
            return 1.0
        denom = max(len(df), len(expected_df), 1)
        diff = abs(len(df) - len(expected_df))
        return max(0.0, 1.0 - (diff / denom))

    def _content_score(self, df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        try:
            df_cmp = self._canon_df(df, loose=True)
            exp_cmp = self._canon_df(expected_df, loose=True)

            common_cols = [c for c in exp_cmp.columns if c in df_cmp.columns]
            if not common_cols:
                return 0.0

            df_sel = df_cmp[common_cols].copy()
            exp_sel = exp_cmp[common_cols].copy()

            for col in common_cols:
                exp_dtype = exp_sel[col].dtype
                if pd.api.types.is_numeric_dtype(exp_dtype):
                    df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")
                    exp_sel[col] = pd.to_numeric(exp_sel[col], errors="coerce")
                else:
                    df_sel[col] = df_sel[col].astype(str).str.strip().str.lower()
                    exp_sel[col] = exp_sel[col].astype(str).str.strip().str.lower()

            exp_counts = Counter(tuple(row) for row in exp_sel.itertuples(index=False, name=None))
            df_counts = Counter(tuple(row) for row in df_sel.itertuples(index=False, name=None))

            matched = sum(min(df_counts[row], exp_counts[row]) for row in exp_counts)

            n_expected = len(exp_sel)
            n_agent = len(df_sel)

            if n_expected == 0:
                return 1.0 if n_agent == 0 else 0.0
            if n_agent == 0:
                return 0.0

            recall = matched / n_expected
            precision = matched / n_agent

            if precision + recall == 0:
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            coverage = len(common_cols) / max(1, len(exp_cmp.columns))

            return min(1.0, f1 * coverage)
        except Exception:
            return 0.0

    def _schema_score(
        self,
        df: pd.DataFrame,
        expected_df: pd.DataFrame,
        constraints: Dict[str, Any],
    ) -> float:
        df_raw = self._canon_df(df, loose=False)
        exp_raw = self._canon_df(expected_df, loose=False)

        df_loose = self._canon_df(df, loose=True)
        exp_loose = self._canon_df(expected_df, loose=True)

        expected_raw = list(exp_raw.columns)
        agent_raw = list(df_raw.columns)

        expected_loose = list(exp_loose.columns)
        agent_loose = list(df_loose.columns)

        if not expected_raw and not agent_raw:
            return 1.0
        if not expected_raw or not agent_raw:
            return 0.0

        def jaccard(a, b):
            sa, sb = set(a), set(b)
            if not sa and not sb:
                return 1.0
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        strict_j = jaccard(expected_raw, agent_raw)
        loose_j = jaccard(expected_loose, agent_loose)

        schema = 0.75 * strict_j + 0.25 * loose_j

        order_bonus = 0.0
        if constraints.get("expected_col_order", False) and agent_raw == expected_raw:
            order_bonus = 0.2

        return min(1.0, 0.85 * schema + order_bonus)

    def _validity_score(
        self,
        df: pd.DataFrame,
        expected_df: pd.DataFrame,
        constraints: Dict[str, Any],
    ) -> float:
        if len(df) == 0:
            return 0.0

        df_loose = self._canon_df(df, loose=True)
        exp_loose = self._canon_df(expected_df, loose=True)
        checks = []

        for col in constraints.get("no_null_cols", []):
            c = self._loose_col(col)
            if c in df_loose.columns:
                checks.append(1.0 - (df_loose[c].isnull().sum() / len(df_loose)))
            else:
                checks.append(0.0)

        for col in exp_loose.columns:
            if col in df_loose.columns and pd.api.types.is_numeric_dtype(exp_loose[col]):
                numeric = pd.to_numeric(df_loose[col], errors="coerce")
                checks.append(numeric.notna().sum() / len(df_loose))

        for col in exp_loose.columns:
            if col in df_loose.columns and not pd.api.types.is_numeric_dtype(exp_loose[col]):
                vals = df_loose[col].astype(str)
                clean = vals.str.strip().str.lower()
                checks.append((vals == clean).sum() / len(df_loose))

        return sum(checks) / len(checks) if checks else 0.0

    def _constraint_score(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> float:
        if len(df) == 0:
            return 0.0

        df_loose = self._canon_df(df, loose=True)
        checks = []

        for col in constraints.get("unique_cols", []):
            c = self._loose_col(col)
            if c in df_loose.columns:
                checks.append(df_loose[c].nunique() / len(df_loose))
            else:
                checks.append(0.0)

        for col, (lo, hi) in constraints.get("range_constraints", {}).items():
            c = self._loose_col(col)
            if c in df_loose.columns:
                numeric = pd.to_numeric(df_loose[c], errors="coerce").dropna()
                if len(numeric) > 0:
                    checks.append(((numeric >= lo) & (numeric <= hi)).sum() / len(numeric))
                else:
                    checks.append(0.0)
            else:
                checks.append(0.0)

        expected_cols = constraints.get("expected_cols", [])
        if expected_cols:
            expected_canon = [self._loose_col(c) for c in expected_cols]
            present = sum(1 for c in expected_canon if c in df_loose.columns)
            checks.append(present / len(expected_canon))

        return sum(checks) / len(checks) if checks else 0.0

    def _extra_row_penalty(self, df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        if len(expected_df) == 0:
            return 0.1 if len(df) > 0 else 0.0

        extra = max(0, len(df) - len(expected_df))
        penalty = 0.14 * (extra / len(expected_df))
        return min(0.35, penalty)