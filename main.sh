#!/bin/bash

######################
## FUNCTIONS ##
######################
# Copy config file
copy_config(){
	config="./config/config_mm.ini"  # choose a config file
	cp ${config} ./config.ini
}


# Make simplex noise
run_simplex_noise(){
    # Set input file
    input_dir='inputs/initial_conditions/'
    input_file=${input_dir}'initial_conditions_inputs.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Create temp input file - in CURRENT DIRECTORY
        echo "$header" > ${input_dir}'tmp.csv'
        echo "$line" >> ${input_dir}'tmp.csv'

        # Run python script
        cat ${input_dir}'tmp.csv'
        python3 generate_simplex_noise.py
        # # Add command to log file
        # echo "python3 generate_simplex_noise.py" >> ${file_log}
        echo "############"

    done
    rm ${input_dir}'tmp.csv'
}

# Make random noise
run_random_noise(){

    # Copy .py file to current directory
    cp scripts/generate_random_noise.py .

    # Set input filepath and filename
    input_dir='inputs/initial_conditions/'
    input_file=${input_dir}'initial_conditions_inputs.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Produce a noisy function for every row in the input file
        # Create temp input file - in CURRENT DIRECTORY (i.e. where main.sh is run from)
        echo "$header" > ${input_dir}'tmp.csv'
        echo "$line" >> ${input_dir}'tmp.csv'

        # Run python script
        cat ${input_dir}'tmp.csv'
        python3 generate_random_noise.py
        # # Add command to log file
        # echo "python3 generate_random_noise.py" >> ${file_log}
        echo "############"

    done

    rm ${input_dir}'tmp.csv' generate_random_noise.py
}


# Steady state evolution
run_evolve_steady_state_square_landscapes(){

    # Copy python script from scripts to current directory
    cp scripts/evolve_ss_square_landscape.py .

    # Set model input file
    input_dir='inputs'
    ss_dir='steady_state'
    input_file=${input_dir}/${ss_dir}/'ss_model_runfile.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Make python infile
        echo "$header" > ${input_dir}/${ss_dir}/'tmp_infile.txt'
        echo "$line" >> ${input_dir}/${ss_dir}/'tmp_infile.txt'

        # Run python script
        python3 evolve_ss_square_landscape.py
        echo "############"

    done

    # Remove temp files and copy of .py file
    rm ${input_dir}/${ss_dir}/'tmp_infile.txt'
    rm evolve_ss_square_landscape.py
}


# Steady state evolution
run_evolve_steady_state_escarpment_landscapes(){

    # Copy .py file to current directory
    cp scripts/evolve_ss_escarpment_landscape.py .

    # Set model input file
    input_dir='inputs'
    ss_dir='steady_state'
    input_file=${input_dir}/${ss_dir}/'ss_model_runfile.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Make python infile
        echo "$header" > ${input_dir}/${ss_dir}/'tmp_infile.txt'
        echo "$line" >> ${input_dir}/${ss_dir}/'tmp_infile.txt'

        # Run python script
        python3 evolve_ss_escarpment_landscape.py
        echo "############"

    done

    # Remove temp files
    rm ${input_dir}/${ss_dir}/'tmp_infile.txt'
    rm evolve_ss_escarpment_landscape.py
}

# Steady state evolution
run_evolve_steady_state_gaussian_landscapes(){

    # Copy .py file to current directory
    cp scripts/evolve_ss_gaussian_landscape.py .

    # Set model input file
    input_dir='inputs'
    ss_dir='steady_state'
    input_file=${input_dir}/${ss_dir}/'ss_model_runfile.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Make python infile
        echo "$header" > ${input_dir}/${ss_dir}/'tmp_infile.txt'
        echo "$line" >> ${input_dir}/${ss_dir}/'tmp_infile.txt'

        # Run python script
        python3 evolve_ss_gaussian_landscape.py
        echo "############"

    done

    # Remove temp files
    rm ${input_dir}/${ss_dir}/'tmp_infile.txt'
    rm evolve_ss_gaussian_landscape.py
}


run_evolve_steady_state_quenched_landscapes(){

    # Copy .py file to current directory
    cp scripts/evolve_ss_quenched_noise.py .

    # Set model input file
    input_dir='inputs'
    ss_dir='steady_state'
    input_file=${input_dir}/${ss_dir}/'ss_model_runfile.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Make python infile
        echo "$header" > ${input_dir}/${ss_dir}/'tmp_infile.txt'
        echo "$line" >> ${input_dir}/${ss_dir}/'tmp_infile.txt'

        # Run python script
        python3 evolve_ss_quenched_noise.py
        echo "############"

    done

    # Remove temp files
    rm ${input_dir}/${ss_dir}/'tmp_infile.txt'
    rm evolve_ss_quenched_noise.py
}


run_evolve_steady_state_nonquenched_landscapes(){

    # Copy .py file to current directory
    cp scripts/evolve_ss_nonquenched_noise.py .

    # Set model input file
    input_dir='inputs'
    ss_dir='steady_state'
    input_file=${input_dir}/${ss_dir}/'ss_model_runfile.csv'

    # Store header
    header=`head -n 1 ${input_file}`

    # Read input file, excluding header
    tail -n +2 $input_file | while IFS="" read -r line || [ -n "$line" ]
    do
        # Make python infile
        echo "$header" > ${input_dir}/${ss_dir}/'tmp_infile.txt'
        echo "$line" >> ${input_dir}/${ss_dir}/'tmp_infile.txt'

        # Run python script
        python3 evolve_ss_nonquenched_noise.py
        echo "############"

    done

    # Remove temp files
    rm ${input_dir}/${ss_dir}/'tmp_infile.txt'
    rm evolve_ss_nonquenched_noise.py
}

#######################

################
## Run things ##
################
copy_config

# Generate noisy initial conditions
#run_random_noise

# Run a model
run_evolve_steady_state_square_landscapes
# run_evolve_steady_state_escarpment_landscapes
# run_evolve_steady_state_gaussian_landscapes
# run_evolve_steady_state_quenched_landscapes
# run_evolve_steady_state_nonquenched_landscapes

# Remove files not needed
rm config.ini
rm -rf scripts/__pycache__