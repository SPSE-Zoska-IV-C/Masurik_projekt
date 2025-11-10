#!/bin/bash

# Find the system-level GNU Radio installation

GNURADIO_PATH=$(python3 -c "import gnuradio; print(gnuradio.__path__[0])")
PMT_PATH=$(python3 -c "import pmt; print(pmt.__path__[0])")

# Identify the Python version used within the virtual environment

PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")

# Create a symbolic link to system-level GNU Radio installation

ln -s $GNURADIO_PATH .venv/lib/python${PYTHON_VERSION}/site-packages/
ln -s $PMT_PATH .venv/lib/python${PYTHON_VERSION}/site-packages/

# Print a success message
echo "Successfully created symbolic link for gnuradio in venv."

deactivate
