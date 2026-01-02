# SNAPY Doxygen Documentation

This directory contains the Doxygen configuration for generating API documentation for the SNAPY project.

## Overview

SNAPY (Simulation of Nonhydrostatic Atmospheres using PyTorch) is a C++ library with PyTorch integration for atmospheric dynamics simulations.

## Prerequisites

To build the documentation, you need:

- **Doxygen** (version 1.8.0 or later)
  - Install on Ubuntu/Debian: `sudo apt-get install doxygen`
  - Install on macOS: `brew install doxygen`
  - Install on Windows: Download from [doxygen.nl](https://www.doxygen.nl/download.html)

- **Graphviz** (optional, for generating diagrams)
  - Install on Ubuntu/Debian: `sudo apt-get install graphviz`
  - Install on macOS: `brew install graphviz`
  - Install on Windows: Download from [graphviz.org](https://graphviz.org/download/)

## Building the Documentation

### Quick Start

From this directory (`docs/doxygen`), run:

```bash
doxygen Doxyfile
```

The generated HTML documentation will be in `output/html/`.

### Opening the Documentation

After building, open the documentation in your browser:

```bash
# On Linux
xdg-open output/html/index.html

# On macOS
open output/html/index.html

# On Windows
start output/html/index.html
```

## Configuration

The documentation is configured via the `Doxyfile` configuration file. Key settings:

- **INPUT**: Set to `../../src` to process all C++ source files
- **RECURSIVE**: YES to process subdirectories
- **EXTRACT_ALL**: YES to document all entities, even those without docstrings
- **GENERATE_HTML**: YES (enabled)
- **GENERATE_LATEX**: NO (disabled by default)
- **SOURCE_BROWSER**: YES to include source code browsing
- **HAVE_DOT**: YES to generate class diagrams (requires Graphviz)

### Excluding Files

Files matching these patterns are excluded from documentation:
- `*/z.junk/*` - Temporary/junk directories
- `*/_*.cpp`, `*/_*.hpp`, `*/_*.h` - Files starting with underscore

## Output

The generated documentation includes:

1. **Class Documentation**: All classes with their members, inheritance, and relationships
2. **File Documentation**: Documentation for each source file
3. **Namespace Documentation**: Organization of code by namespace
4. **Class Diagrams**: Inheritance and collaboration diagrams (if Graphviz is available)
5. **Source Code Browser**: Browse the actual source code with cross-references

## Customization

To customize the documentation:

1. Edit `Doxyfile` to change configuration options
2. Modify docstrings in the source code (use Doxygen style with `//!` or `/** */`)
3. Add a project logo by setting `PROJECT_LOGO` in Doxyfile
4. Customize HTML output with custom CSS via `HTML_EXTRA_STYLESHEET`

## Docstring Style

SNAPY uses Doxygen-style docstrings. Example:

```cpp
//! \brief Brief description of the function
//!
//! Detailed description of what the function does,
//! including any important notes or caveats.
//!
//! \param[in] input Input parameter description
//! \param[out] output Output parameter description
//! \return Description of return value
//!
//! \note Additional notes
//! \warning Important warnings
torch::Tensor my_function(torch::Tensor input, torch::Tensor& output);
```

## Troubleshooting

### Doxygen not found

Make sure Doxygen is installed and in your PATH:
```bash
doxygen --version
```

### No class diagrams

Install Graphviz and ensure `dot` is in your PATH:
```bash
dot -V
```

### Warnings about undocumented entities

This is normal. The documentation extracts all code, but warnings help identify
what still needs documentation. To suppress these warnings, set:
```
WARN_IF_UNDOCUMENTED = NO
```

## Continuous Integration

To integrate documentation generation into CI/CD:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y doxygen graphviz

# Build docs
cd docs/doxygen
doxygen Doxyfile

# Optionally deploy to GitHub Pages or similar
```

## Additional Resources

- [Doxygen Manual](https://www.doxygen.nl/manual/)
- [Doxygen Special Commands](https://www.doxygen.nl/manual/commands.html)
- [Documenting C++ Code](https://www.doxygen.nl/manual/docblocks.html)

## License

The documentation follows the same license as the SNAPY project.
