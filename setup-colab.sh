#!/bin/bash
# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    sudo apt-get install ffmpeg
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