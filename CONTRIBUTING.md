# Contributing

This research project studies DPO-based compression of mathematical reasoning.
Focused bug fixes, regression tests, documentation corrections, and reproducible
evaluation reports are useful contributions.

Start with the [README](README.md) for scripts and configurations and the
[portfolio overview](docs/PORTFOLIO.md) for recorded results and their limits.
Search existing issues and pull requests before opening a new one. Discuss large
training runs or changes to the evaluation protocol in an issue first.

## Report a problem

Use the bug report form for code failures and the evaluation report form for
questions about metrics or reproducibility. Include the commit, relevant runtime
versions, exact command and configuration, and expected versus actual behavior.
Prefer a small synthetic input that others can run without model downloads,
credentials, or a GPU. If a check was not run, say so and explain what is missing.

Keep evidence tied to its scope. For evaluation reports, identify the dataset and
split, model revision, prompt and decoding settings, whether the data influenced
checkpoint selection, and the metric's unit. Whitespace-separated words and
tokenizer completion tokens are different measures. A passing code check alone
does not establish model quality or inference speed.

## Prepare a pull request

1. Create a branch for one focused change and link any related issue.
2. Describe the failure or motivation and the resulting behavior.
3. Add a small regression check for a behavior change when practical. For docs,
   verify referenced commands, paths, and claims against the source.
4. List checks actually run, their results, and any unverified behavior. Do not
   present a proposed experiment as a completed result.

From the repository root, run the existing lightweight check:

```bash
python3 -S scripts/03_evaluate_generations.py --self_test
git diff --check
```

The self-test uses the Python standard library to check answer extraction and
JSONL reading/writing. It does not install dependencies, load a model, call an
inference API, or validate the training pipeline. Further checks should match the
change; GPU training and paid inference are not prerequisites for a contribution.
The README contains the separate setup and experiment commands.

## Keep shared artifacts reviewable

Do not commit credentials, personal information, model weights, generated
datasets, or raw generation outputs. Use small synthetic examples and sanitized
logs instead. The repository's ignored data and output directories are local
working areas, not a request to upload their contents.

Keep historical results and their provenance intact. If an experiment changes a
split, prompt, model revision, or length definition, report it as a separate
comparison and explain its limits. Follow the locked evaluation protocol rather
than tuning it after inspecting the results.
