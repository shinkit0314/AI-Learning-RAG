# AI-Learning-RAG agent guidance

## Environment

- The project uses Python 3.12 and uv. `uv.lock` is the reproducible dependency baseline.
- Do not automatically install or synchronize the environment unless the user explicitly asks you to do it.
- At the beginning of a new conversation that needs Python, explain the current project stage and remind the user to enter the repository directory first.
- Before asking the user to run an environment command, explain which step it belongs to, what it changes, and what successful output should look like.
- On Windows, guide the user to run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1`. On other systems, guide the user to run `uv sync --locked --dev`.
- Wait for the user to share or confirm the result, then help diagnose any error before continuing.
- Run project tools through `uv run`, for example `uv run pytest` and `uv run ruff check .`.
- Use `docs/ENVIRONMENT_SETUP.md` as the beginner-facing environment guide.

## Quality gate

- Add or update tests with every behavior change.
- Before handing work back, run `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest`.
- Do not treat an iteration as accepted until automated checks pass and the user confirms it manually.

## Teaching approach

- Assume the user is learning RAG from the beginning. Explain the current stage, the purpose of a command, and the expected result before asking the user to run it.
- When the user is unsure what to do next, first provide orientation and a small decision framework. Recommend a low-risk next step when needed and explain the reasoning.

## File safety

- Never delete files or directories recursively.
- Delete only one explicitly identified file at a time. If multiple files need deletion, stop and ask the user to remove them manually.
