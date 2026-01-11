# Snapy Documentation

This directory contains the Sphinx-based documentation for Snapy that is built on ReadTheDocs.

## Building the Documentation

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Build Locally

From this directory, run:

```bash
sphinx-build -b html . _build/html
```

Then open `_build/html/index.html` in your browser.

Alternatively, you can use:

```bash
make html  # If you have make installed
```

### Watch for Changes

For live reload during development:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── examples.rst           # Example code and use cases
├── api/                   # API reference documentation
│   ├── index.rst
│   ├── mesh.rst          # Core modules
│   ├── hydro.rst
│   ├── eos.rst
│   ├── coordinate.rst
│   ├── boundary.rst
│   ├── output.rst
│   ├── reconstruction.rst
│   ├── riemann.rst
│   ├── forcing.rst
│   ├── implicit.rst
│   ├── layout.rst
│   └── utilities/        # Utility modules
│       ├── pd_inspect.rst
│       ├── pd_combine.rst
│       ├── exchange.rst
│       └── write_init.rst
└── user_guide/           # Detailed user guides
    ├── index.rst
    ├── configuration.rst  # Configuration reference
    ├── simulation.rst     # Running simulations
    ├── distributed.rst    # Distributed computing
    └── output.rst         # Output and postprocessing
```

## ReadTheDocs Integration

The documentation is automatically built on ReadTheDocs when changes are pushed to the repository. The configuration is in:

- `.readthedocs.yaml` - ReadTheDocs build configuration
- `docs/conf.py` - Sphinx configuration
- `docs/requirements.txt` - Python dependencies for building docs

### ReadTheDocs URL

The documentation is hosted at: https://snapy.readthedocs.io

## Updating Documentation

### Adding New Modules

1. Create a new `.rst` file in the appropriate directory (`api/` or `user_guide/`)
2. Add the file to the `toctree` in the relevant `index.rst`
3. Build locally to verify
4. Commit and push

### Documentation Style

- Use reStructuredText (`.rst`) format
- Follow the existing structure and style
- Include code examples where appropriate
- Document all parameters, return values, and types
- Add cross-references to related modules using `:doc:` or `:ref:`

### Python API Documentation

The Python API is documented based on the `.pyi` stub files in `python/snapy/`. These stub files provide type hints and are the source of truth for the API.

## Sphinx Extensions

The documentation uses these Sphinx extensions:

- `sphinx.ext.autodoc` - Auto-generate documentation from docstrings
- `sphinx.ext.napoleon` - Support for Google/NumPy style docstrings
- `sphinx.ext.viewcode` - Add links to source code
- `sphinx.ext.intersphinx` - Link to other project documentation
- `sphinx_autodoc_typehints` - Add type hints to documentation
- `myst_parser` - Support for Markdown files

## Theme

The documentation uses the Read the Docs theme (`sphinx_rtd_theme`). Theme configuration is in `conf.py`.

## Doxygen Documentation

The `doxygen/` subdirectory contains C++ API documentation built with Doxygen. This is separate from the Sphinx-based Python documentation.

## Troubleshooting

### Build Warnings

Some warnings are expected:
- Intersphinx warnings about external inventories (when building offline)
- Duplicate module warnings (these are intentional for organization)

### Missing Dependencies

If you get import errors during build:

```bash
pip install -r requirements.txt
```

### Broken Links

Check for broken links:

```bash
sphinx-build -b linkcheck . _build/linkcheck
```

## Contributing

When contributing to documentation:

1. Build locally first to check for errors
2. Follow the existing style and structure
3. Add examples where helpful
4. Keep descriptions clear and concise
5. Test any code examples to ensure they work

## License

The documentation follows the same license as the Snapy project (MIT License).
