#!/bin/bash

# Install FFMPEG
os=$(uname -a)

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    if [[ $os == *"arch"* ]]; then
        # If the OS is Arch Linux, use pacman
        sudo pacman -S ffmpeg
    elif [[ $os == *"Ubuntu"* ]] || [[ $os == *"Debian"* ]]; then
        # If the OS is Ubuntu or Debian, use apt-get. Also install python3-venv because it is not installed by default in debian.
        sudo apt-get install ffmpeg
        sudo apt install python3-venv
    fi
fi

# Check if CUDA is installed
if [ -z "$CUDA_PATH" ]
then
    echo "CUDA is not installed. Please install CUDA."
    echo "CPU will be used for inference."
fi

# Create a virtual environment
python3 -m venv .lip-wise

wait

# Activate the virtual environment
source .lip-wise/bin/activate

wait

# Get Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

# Install requirements
pip install -r requirements.txt

# Copy archs
cp archs/* .lip-wise/lib/python${python_version}/site-packages/basicsr/archs/

# Run file_check.py
python ./helpers/file_check.py