# recommendation-system

A Python project for the class I'm giving on Big Data & Artificial Intelligence for the specific course on Recommendation System. Keep being under construction with every class.

---

## 🚀 Installation

You can install this project using [`uv`](https://github.com/astral-sh/uv):

```sh
uv sync --no-dev
```

## 📂 Project Structure

```
recommendation-system/
├── scripts/                 # Executable scripts to run package functionalities
├── recommendation_system/      # Main package
│   ├── __init__.py
│   └── utils                # Sub-package for utils modules
│       ├── env.py           # Environment related utils
│       └── determinism.py   # Seeding and random generation utils
├── tests/                   # Pytest suite
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## 💻 Development

To start developing some new stuff, first install all dependencies and pre-commit hooks:

```sh
uv sync
uv run pre-commit install
```

In order to add a new dependency be sure to do so with `uv` by running:

```sh
uv add <python-package-name>

```

And to run a python script or module, you can simply use:

```sh
uv run <path-to-python-file>
```

## 🧪 Running Tests

To run the test suite:

```sh
uv run pre-commit run --all-files
uv run mypy
uv run pytest
```

## 📚 Documentation

The documentation of the package is generated using MkDocs.
To serve the docs locally, install the required dependencies first:

```sh
uv sync --group docs
```

and run the following command:

```sh
uv run mkdocs serve
```

This will run a development server where you can view the documentation
and see changes in real-time as you edit the docs files.

---

## 👤 Maintainers

- **Èric Quintana Aguasca** - [saericquintana@gmail.com](mailto:saericquintana@gmail.com)
