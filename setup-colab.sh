#!/bin/bash
# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    sudo apt-get install ffmpeg
fi
# Get python version:
python_version=$(python -v)

# Check if python3-venv is installed
if ! dpkg-query -W -f='${Status}' python${python_verison}-venv 2>/dev/null | grep -q "ok installed"; 
then
  sudo apt install python${python_verison}-venv
fi

# Check if CUDA is installed
if [ -z "$CUDA_PATH" ]
then
    echo "CUDA is not installed. Please install CUDA."
    echo "CPU will be used for inference."
fi

# Install requirements
pip install -r requirements.txt
pip install --upgrade --no-cache-dir gdown

# Copy archs
cp archs/* /usr/local/lib/python3.10/dist-packages/basicsr/archs

# Run file_check.py
python ./helpers/file_check.py