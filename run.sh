#!/bin/bash

# Define a list of files to iterate over
files=(918K 459K 393K 229K 197K 115K 98K)

# Use a for loop to iterate over the list
for file in "${files[@]}"
do
    echo "$file"
    python run_simba.py --model "$file" --pixel_attack --freq_dims 64
done

python load.py --pixel_attack