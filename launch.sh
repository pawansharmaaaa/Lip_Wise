#!/bin/bash
for arg in "$@"
do
    if [ "$arg" == "--colab" ]
    then
        python launch.py --colab
        exit 0
    else
        source .lip-wise/bin/activate
        python launch.py
    fi
done