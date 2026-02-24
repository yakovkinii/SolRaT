1. Bump version.
2. Run tests: 
    ```bash 
    python run_all_tests.py
    ```
3. Run coverage:
    ```bash
    coverage run -m unittest discover
    ```
    ```bash
    coverage report
    ```
    ```bash
    coverage html
    ```
    ```bash
    rm .coverage
    ```
    ```bash
    rm -r htmlcov
    ```
4. Build docs:
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
5. Build and upload package
    ```bash
    pip install build twine
    ```
    ```bash
    rm -r dist
    ```
    ```bash
    python -m build
    ```
    ```bash
    twine upload dist/*
    ```
    ```bash
    rm -r dist
    ```
    ```bash
    rm -r solrat.egg-info
    ```