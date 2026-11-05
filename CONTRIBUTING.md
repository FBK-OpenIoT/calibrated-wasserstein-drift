# Contributing

Thank you for considering a contribution. Bug reports, documentation fixes,
tests, and focused improvements to the detector or research workflows are
welcome.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```bash
uv sync --extra research --extra notebook --group dev
```

The base installation intentionally contains only the dependencies required by
`WassersteinDriftDetector`. The extras provide the evaluation pipeline,
experiments, plots, and notebook environment.

## Checks

Before opening a pull request, run:

```bash
uv run ruff check .
uv run pytest --cov=wasserstein_score
uv build
uvx twine check dist/*
```

New behavior should include tests. Avoid committing downloaded datasets,
generated results, notebook output, caches, credentials, or machine-specific
paths; generated results belong under the Gitignored `data/` directory.

## Pull requests

Keep changes focused and describe:

- the problem being solved;
- the approach and any methodological assumptions;
- how the change was verified;
- whether public behavior or experiment results change.

Research changes should preserve deterministic seeds where possible and state
the expected runtime and input data.
