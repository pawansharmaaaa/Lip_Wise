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
        # If the OS is Ubuntu or Debian, use apt-get
        sudo apt-get install ffmpeg
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

# Activate the virtual environment
source .lip-wise/bin/activate

# Install requirements
pip install -r requirements.txt