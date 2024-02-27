#!/bin/bash
# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    sudo apt-get install ffmpeg
fi

# Install requirements
pip install -r requirements.txt
pip install --upgrade --no-cache-dir gdown

# Find python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")

# Copy archs
cp archs/* /usr/local/lib/python${python_version}/dist-packages/basicsr/archs

# Run file_check.py
python ./helpers/file_check.py