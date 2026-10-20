# Contributing to ALHENA

Bug reports, questions and suggestions are welcome in the
[issues](https://github.com/prism-oncology/ALHENA/issues). Include the command or code you ran, the
message it printed in full, and what `alhena info` reports.

## Development install

Clone the repository and install it in editable mode, with the development tools:

```bash
git clone https://github.com/prism-oncology/ALHENA.git
cd ALHENA
pip install -e ".[dev]"
```

Add the `viz`, `wsi` and `spatial` extras to run the tests that need them.

## Tests and style

The tests use synthetic data only, so they need no model weights and no network:

```bash
pytest
```

The code is linted and formatted with ruff, as the continuous integration checks it:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
```

## Documentation

The documentation is built with MkDocs. Install the docs extra, then serve the site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
