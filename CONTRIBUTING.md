# Contributing

Contributions and bug reports are welcome.

1. Keep changes focused and document updates to preprocessing, loading, or evaluation.
2. Never commit raw dataset files, archives, Kaggle credentials, API tokens, virtual environments, or compiled PDFs.
3. Run checks locally before committing:
   - `python -m compileall src tests`
   - `python -m pytest -v`
4. Report only measured results with the exact command and configuration used.
