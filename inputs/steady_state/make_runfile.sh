#!/bin/bash
### Make a ss_model_runfile

# Specify the seed file - USER INPUT
# seedfile='../initial_conditions/random_100_seeds.txt'
echo "1" > example_seedfile.txt
seedfile='example_seedfile.txt'

# Set forward parameter file - USER INPUT
forward_file="inputs/steady_state/forward_model_inputs.csv"

# Set header
header="initial_condition_file, forward_model_input_file"

# Input file prefix
prefix="inputs/initial_conditions"

# Set noise type, random or simplex
noise="random"

# Set amplitude
amp="1.0m"

# Make the file with just header
echo ${header} > ss_model_runfile.csv

# Loop through each colour then each seed in seedfile
for colour in red white blue
do
    while IFS= read -r seed
    do
        echo ${prefix}/${noise}/${colour}_${amp}/${noise}_${colour}_${amp}_${seed}.npy","${forward_file} >> ss_model_runfile.csv
        echo "Added row for ${noise} ${colour} ${amp} ${seed}"
    done < $seedfile
done
