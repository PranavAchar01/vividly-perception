# Thin, discoverable aliases for the commands in README.md and TRAINING.md.
.PHONY: demo sync sync-cuda sync-agpl sync-curate lint fmt guards test smoke hooks lock-check

demo:            # the local model demo platform: upload a video, watch the harness run
	uv run python scripts/run_demo_platform.py

sync:            # macOS development environment, no torch
	uv sync

sync-cuda:       # Linux training and evaluation environment on a GPU
	uv sync --extra cuda --no-group mac --locked

sync-agpl:       # the AGPL reference environment, in its own virtual environment
	UV_PROJECT_ENVIRONMENT=.venv-agpl uv sync --extra eval-agpl --no-default-groups

sync-curate:     # the curation, tracking-metric and export environment
	UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate --no-default-groups

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff check --fix . && uv run ruff format .

guards:
	uv run python scripts/guards/check_no_fisheye.py
	uv run python scripts/guards/check_no_fisheye_primitives.py
	uv run python scripts/guards/check_agpl_imports.py
	uv run python scripts/guards/check_face_deps.py
	uv run python scripts/guards/check_train_data_paths.py
	uv run python scripts/guards/check_config_registry.py
	uv run python scripts/guards/check_reid_weight_sources.py
	uv run lint-imports
	bash scripts/guards/selftest.sh

test:
	uv run pytest -m "not gpu and not slow"

smoke:
	uv run python scripts/eval/smoke_test.py --config configs/eval/smoke_mac.yaml

hooks:
	uv run pre-commit install

lock-check:
	uv lock --check
