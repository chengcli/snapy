#!/bin/bash
# Build script for SNAPY Doxygen documentation

set -e  # Exit on error

echo "======================================"
echo "Building SNAPY Doxygen Documentation"
echo "======================================"
echo

# Check if doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "ERROR: Doxygen is not installed."
    echo "Please install doxygen:"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  macOS: brew install doxygen"
    echo "  Windows: Download from https://www.doxygen.nl/download.html"
    exit 1
fi

echo "Doxygen version: $(doxygen --version)"

# Check if graphviz/dot is available (optional)
if command -v dot &> /dev/null; then
    echo "Graphviz version: $(dot -V 2>&1)"
    echo "Class diagrams will be generated."
else
    echo "WARNING: Graphviz not found. Class diagrams will not be generated."
    echo "Install graphviz for better documentation:"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  macOS: brew install graphviz"
fi

echo
echo "Building documentation..."
echo

# Run doxygen
doxygen Doxyfile

echo
echo "======================================"
echo "Documentation built successfully!"
echo "======================================"
echo
echo "Output location: output/html/"
echo "Open in browser: output/html/index.html"
echo
