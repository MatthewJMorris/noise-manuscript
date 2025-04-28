# noise-manuscript
The code provided in this repository relates to the manuscript entitled "Impact of noise on landscapes and metrics generated with Stream Power models". It is designed to:
1. Provide a library of noisy functions (and associated code to generate such functions) with different spectral properties that can be used within landscape evolution models
2. Provide a basic working example of code used to generate models within the aforementioned publication

Something not clear about how to use this code? Interested in working with me? Email me: matthew.morris15@imperial.ac.uk.

Please respect the user license associated with this code. You should cite the associated manuscript (DOI: XXX) in any research produced using the materials provided herein.

<p align="center">
<img src="https://github.com/user-attachments/assets/b7d0476e-4c60-4bb6-8760-df7e3d2189aa" width=60% height=60%>
</p>

*An example of a typical noisy function with red noise characteristics; spectral power proportional to wavenumber* $k^{-2}$ *, that can be generated from code within this repository.*


## Software Languages and Requisites
The code in this directory is written in Python and bash. Code was generated using an Ubuntu OS and Python v3.10.13. Landscape evolution models were generated using the [Landlab](https://landlab.csdms.io/) surface process package version 2.7.0.

It is recommended to install required software using a package manager, such as conda.

## Workflow
This repository contains 4 directories and a `main.sh` file.

### config
Contains a `.ini` configuration file, used to store directory paths. Used as a way to make code within .py files as generic as possible such that it can be run on different machines by changing only the directories within the config file. There are, of course, plenty of other ways of achieving this.

### inputs
Contains two directories:
1. **initial_conditions**: A library of noisy functions (as .npy files) that can be used within landscape evolution models (e.g. as starting conditions) is contained within this directory. Also stored here are a list of random integer seeds used to generate the noisy functions, and an `initial_conditions_inputs.csv` file, used to define the type of noise, dimensions, amplitude, and random seed.
2. **steady_state**: Contains 5 files used for generating synthetic landscapes. Model parameter values are defined in `forward_model_inputs.csv`. Edit and run (from within this directory) `make_runfile.sh` to generate a runfile for a single (or multiple) models. `ss_model_runfile_paper` is the runfile used to generate models within the paper.

### outputs
Contains a directory path to where models and their associated outputs are saved.

### scripts
Contains a series of scripts used for generating noisy functions and synthetic landscapes.

### main.sh
This file is used to run .py scripts for generating noisy functions and models. For example, to reproduce a small subset of the noisy functions within `inputs/initial_conditions/random`, uncomment the calls to functions `copy_config` and `run_random_noise`, and run `bash main.sh` from your terminal.

## Example Usage
A straightforward way to download the library of noisy functions and associated code for your own usage is to clone this repository using git:

`git clone https://github.com/MatthewJMorris/noise-manuscript.git /path/to/your/directory`

Alternatively, you can click on the repository release and download it using the GitHub interface.
