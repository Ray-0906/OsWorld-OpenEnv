# Multi-Component Semantic Grader

Unlike traditional strict binary grading, this environment implements an autonomous **Multi-Component Semantic Grader** designed to give smooth partial credit at every stage of cleaning.

## The Multi-Component Architecture

The grader computes a composite score ($\Phi$) based on content, structure, and rule enforcement:

```text
structural = 0.35 * schema_score + 0.30 * validity_score + 0.35 * constraint_score
gated_content = content_score * (0.6 + 0.4 * schema_score)
base = (0.55 * gated_content + 0.45 * structural) * row_balance
Phi = base ** 1.05 - extra_row_penalty
```

The $\Phi$ score is clamped to **[0.0001, 0.9999]**. The task is officially solved at $\Phi \geq 0.99$.

### Component Breakdown

#### 1. Content Score (55% weight via gated blend)
F1-based multiset row matching using loose column normalization. It can handle case, whitespace, and underscore variants (e.g. treating `"ID"` and `"id"` equally). Its coverage factor penalizes missing columns naturally.

#### 2. Schema Score
Blended strict + loose Jaccard similarity (75% strict, 25% loose) across expected column names. 
- **Bonus:** +0.2 awarded if columns match the exact expected order.

#### 3. Validity Score (30% structural weight)
Evaluates data formatting rules:
- Null rate in required columns
- Numeric type correctness
- String formatting (must be stripped and lowercased)

#### 4. Constraint Score (35% structural weight)
Verifies domain-specific logical guidelines:
- Uniqueness of key columns
- Value range enforcement
- Required column presence

#### 5. Row Balance Multiplier
Penalizes row count mismatches simultaneously against all composite scores:
```text
row_balance = max(0.0, 1.0 - abs(n_agent - n_expected) / max(n_agent, n_expected))
```

#### 6. Anti-Cheat Penalty 
To block exploits like duplicate abuse, partial limits, and inflation:
```text
penalty = min(0.35, 0.14 * max(0, n_agent - n_expected) / n_expected)
```