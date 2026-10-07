# Selected skills from ASkills

This file records the project-relevant skills selected from the read-only
reference collection at `C:\Users\Admin\Desktop\ASkills`.

Only project-local, condensed adaptations are stored in `.agents/skills/`.
The source collection is reference material and must not be modified.

## Selected skills

| Project skill | Source skill | Why it fits this project |
| --- | --- | --- |
| `python-project-scaffold` | `skills/python-development-python-scaffold` | Keeps the research codebase maintainable with `uv`, `src/`, configuration, scripts, and tests. |
| `python-packaging` | `skills/python-packaging` | Guides `pyproject.toml`, package discovery, dependencies, and reproducible builds. |
| `hugging-face-cli` | `skills/hugging-face-cli` | Adds safe Hub authentication, cache, model, and dataset operation guidance. |
| `hugging-face-evaluation` | `skills/hugging-face-community-evals` | Provides local evaluation backend selection and smoke-test discipline. |
| `vlm-research` | `skills/computer-vision-expert` | Keeps the project focused on VLM/VQA and visual-token research; unrelated detection/segmentation material was omitted. |
| `model-memory-planning` | `skills/hf-mem` | Supports memory estimation before loading Qwen3-VL or scaling experiments. |
| `research-decision-records` | `skills/architecture-decision-records` | Captures method and protocol decisions while adaptive pruning is still open. |
| `python-testing` | `skills/python-testing-patterns` | Establishes a lightweight pytest strategy for loaders, configs, and evaluation logic. |

## Not copied

The collection contains many skills for web applications, agents, deployment,
JavaScript, databases, and unrelated computer-vision tasks. They were not
copied because they do not help the current dataset-download plus Qwen3-VL
baseline milestone.

The project already has dedicated local skills for `uv-python`,
`huggingface-datasets`, and `git-workflow`; those remain authoritative for
project-specific constraints.
