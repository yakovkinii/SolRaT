> Manual pre-release checklist. Automated quality control (formatting, linting, tests, coverage) is enforced by CI on all PRs and merges.

1. Bump version.
2. Run formatters:
    ```bash 
    isort .
    ```
    ```bash 
    black .
    ```
3. Run pre-release checks: 
    ```bash 
    isort --check-only .
    ```
    ```bash 
    black --check .
    ```
    ```bash 
    flake8 .
    ```
    ```bash 
    coverage run -m pytest
    ```
    ```bash 
    coverage report
    ```
    ```bash 
    python -m build
    ```
4. Check and update docs:
    ```bash
    pip install -e .
    ```
    ```bash
    rm -r docs/build/*
    ```
    ```bash
    rm docs/source/solrat*
    ```
    ```bash
    sphinx-apidoc -o .\docs\source\ ./solrat -M -e -T
    ```
    ```bash
    rm docs/source/solrat.rst
    rm docs/source/solrat.about.rst
    rm docs/source/solrat.atom_model.rst
    ```
    ```bash
    sphinx-build -M html docs/source/ docs/build/
    ```
    ```bash
    rm -r docs/build/*
    ```
    ```bash
    pip uninstall solrat
    ```
    ```bash
    rm -r solrat.egg-info
    ```
5. Clean up:
    ```bash
    rm -r build/*
    ```
    ```bash
    rm -r docs/build/*
    ```
    ```bash
    rm -r dist
    ```
    ```bash
    rm -r solrat.egg-info
    ```