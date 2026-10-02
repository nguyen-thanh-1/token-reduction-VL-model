---
name: git-workflow
description: Keep repository changes reviewable and safe.
---

# Git workflow

- Inspect `git status`, the current branch, and the remote before changing
  repository state.
- Develop on `dev` unless the user specifies another branch.
- Never commit `.venv`, uv caches, downloaded datasets, credentials, or local
  machine configuration.
- Keep commits focused and use imperative messages when the user asks for a
  commit.
- Do not push, open pull requests, or change remote branches unless the user
  explicitly asks for that external action.
- Before handoff, report changed files, validation performed, and whether
  changes were committed or pushed.

Do not use destructive commands such as `git reset --hard` or `git clean -fd`
without explicit user authorization. Preserve unrelated working-tree changes.
