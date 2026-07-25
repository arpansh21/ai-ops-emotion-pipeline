# AI Skill Publish Workflow

## Overview

This repository includes a GitHub Actions workflow that:
- runs tests for AI workflow monitoring
- analyzes the repository for AI-related content and issue signals
- publishes a GitHub release with a model artifact when the `main` branch is updated

The setup is designed for a private GitHub account and focuses on AI/agent/skill workflows.

## Workflow orchestration

The workflow file is `.github/workflows/ai-skill-publish.yml`.

### Trigger events
- `push` to `main`
- `workflow_dispatch` (manual run)
- scheduled daily run at `02:00 UTC`

### Publish pipeline jobs

1. **test**
   - checks out the repository
   - sets up Python 3.11
   - installs dev dependencies from `requirements-dev.txt`
   - runs `pytest tests`

2. **analyze**
   - depends on `test`
   - regenerates the AI Ops analysis report
   - produces markdown and JSON intelligence outputs
   - uploads `ai_ops_report.md` as an artifact

3. **validate**
   - depends on `analyze`
   - validates the presence of `model.h5` and `ai_ops_report.md`
   - packages model and report assets into `release/ai-ops-release.zip`
   - uploads the validation package as an artifact

4. **publish**
   - depends on `validate`
   - reruns the AI Ops quality gate with `--fail-on-issue`
   - creates a GitHub release automatically on `main`
   - attaches `model.h5` and `ai-ops-release.zip` to the release

5. **monitor**
   - runs on schedule and manual trigger
   - generates periodic AI Ops health reports
   - uploads `ai_ops_monitor.md` as an artifact

6. **summary**
   - reports workflow completion for scheduled monitoring runs
   - confirms monitor artifacts are available for review

## What this publishes

The pipeline now publishes:
- a GitHub release tagged automatically from the workflow run number
- `model.h5` as a release asset
- `ai-ops-release.zip` containing the model and AI Ops report
- scheduled monitor report artifacts for continuous AI Ops review

## AI Ops agent

`ai_ops_agent.py` is the central AI Ops evaluator.

It now checks for:
- training data structure, class balance, and empty categories
- saved model artifact and label metadata health
- test data coverage and train/test drift
- prompt/workflow documentation and AI pipeline awareness
- repository documentation coverage for AI Ops topics
- high-risk secret patterns in code and docs
- issue-related signals like `error`, `exception`, `timeout`, `todo`, `drift`, and `bias`

The script generates both markdown and JSON reports, and the pipeline can fail when real-time risk findings occur.

## Tests

The test suite is now in `tests/test_ai_ops_agent.py`.

It verifies:
- AI Ops skill detections for missing train/test assets
- model artifact issue handling
- issue keyword detection
- CLI report generation

## Local usage

Run tests locally:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Run the analyzer locally:

```bash
python ai_ops_agent.py --root . --output ai_ops_report.md --output-json ai_ops_report.json
```

Fail when issues are present:

```bash
python ai_ops_agent.py --root . --output ai_ops_report.md --output-json ai_ops_report.json --fail-on-issue
```

## Notes

- `GITHUB_TOKEN` is used automatically by GitHub Actions for release creation.
- Release artifacts include `model.h5`.
- The workflow is intended to catch AI-related content and operational issues automatically.
