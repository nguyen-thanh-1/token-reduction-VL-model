---
name: research-decision-records
description: Record major research and architecture decisions while the pruning method is still open.
source: C:\Users\Admin\Desktop\ASkills\skills\architecture-decision-records\SKILL.md
---

# Research decision records

Create a short ADR when choosing or changing a model, dataset protocol,
pruning location, token-scoring signal, budget policy, metric, or evaluation
backend.

Each record should contain:

1. Context and decision drivers.
2. Options considered and their trade-offs.
3. The selected decision and rationale.
4. Consequences, risks, and a validation plan.
5. Links to affected configs, scripts, metrics, and related records.

Do not rewrite an accepted decision to hide history. Supersede it with a new
record when later experiments invalidate it. Keep early records marked
`Proposed` when the method is not yet locked.
