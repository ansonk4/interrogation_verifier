# Repository Guidelines

## Coding Style & Naming Conventions

This repo is an MVP: keep code minimal, clean, and centered on the core graph-verification artifact. Outputs and saved files should be concise, direct, easy to understand, and easy to inspect; avoid verbose logs, decorative formatting, or extra files.

Keep configuration and command-line arguments concise. Add an option only when it changes useful behavior now. Remove unused flags, dead helpers, broad abstractions, and boilerplate as soon as they stop carrying their weight. Keep generated files, build artifacts, and virtual environments out of version control.

No formatter or linter is configured yet. If one is introduced, add its configuration to `pyproject.toml` and document the exact `uv run ...` command here.

## Agent-Specific Instructions

Before modifying repository guidance, check whether `AGENTS.md` already exists and preserve user changes. Keep future instructions specific to this repository.


## LLM Endpoint Selection
When code needs to use an LLM endpoint, unless otherwise specified, use `model/openrouter/hy3.json` as the default model.
