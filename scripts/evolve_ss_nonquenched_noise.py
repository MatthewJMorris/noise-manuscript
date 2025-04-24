"""
Evolve a coloured landscape to steady state whilst feeding in random
noise at each forward model timestep.
"""
# Load required packages
import os
import shutil
import warnings
from configparser import ConfigParser
import numpy as np
import pandas as pd
import _pickle as pickle
from landlab.components import (ChannelProfiler,
                                ChiFinder,
                                HackCalculator)
from landlab.io.native_landlab import save_grid
from scripts.landscape_functions import initialise_a_model
from scripts.geomorphic_analysis_functions import (extract_chi_profile_data,
                                                   extract_hack_data)

# Load config file and extract directories
config_obj = ConfigParser()
config_obj.read('config.ini')
dirs = config_obj["directories"]
# Input data directories
curr_dir = dirs["base"]
data_input_main = dirs["data_input"]
data_input_ic = os.path.join(data_input_main, dirs["input_initial_conditions"])
data_input_ss = os.path.join(data_input_main, dirs["steadystate"])
# Output data directories
data_output_main = dirs["data_output"]
data_output_ss = os.path.join(data_output_main, dirs["steadystate"])
data_output_nonquenched = os.path.join(data_output_ss, dirs["nonquenched"])

# Read infile to get inputs
infile = np.loadtxt(data_input_ss + 'tmp_infile.txt', delimiter=',', dtype='str', skiprows=1)
ic_filepath = os.path.join(curr_dir, infile[0].strip())
forward_filepath = os.path.join(curr_dir, infile[1].strip())

# Extract IC file information to get the seed and save directory
ic_file = ic_filepath.split("/")[-1].split("_")
colour = ic_file[1]
amplitude = float(ic_file[2].split('m')[0])
seed = ic_file[-1][:-4]  # last part of filename, all but last 4 chars
print(f"### Seed = {seed} ###")
save_filepath = ic_filepath.split("/")[-3:-1]
save_filepath = "/".join(save_filepath)
data_savedir = os.path.join(data_output_nonquenched, save_filepath)
if not os.path.exists(data_savedir):
    os.makedirs(data_savedir + '/figs')
    os.makedirs(data_savedir + '/inputs')
    os.makedirs(data_savedir + '/ss_info')
    os.makedirs(data_savedir + '/grids')
    os.makedirs(data_savedir + '/profilers')

# Load initial condition file
print("### Loading initial condition ###")
start_topo = np.load(ic_filepath)
# Extract dimensions
nx = start_topo.shape[1]
ny = start_topo.shape[0]
dx = 1000  # grid spacing in metres

# Load erosional input file
print("### Initialising grid and forward model ###")
# Extract forward model parameter values from file
df = pd.read_csv(forward_filepath,
                     dtype={'runtime_kyr': float, 'n_sp': float, 'm_sp': float, 'K_sp': float,
                            'D_diff': float}, skipinitialspace=True).to_dict(orient='records')
n_sp = df[0]['n_sp']
m_sp = df[0]['m_sp']
K_sp = df[0]['K_sp']
D_diff = df[0]['D_diff']
# Model dimensions
dims = {'ny': 100,
        'nx': 100,
        'dx': 1000}
# Forward model parameter values
model_param_vals = {'K_sp': K_sp,
        'n_sp': n_sp,
        'm_sp': m_sp,
        'D_diff': D_diff}
# Boundary conditions
boundaries = {'right': False,
             'top': False,
             'left': False,
             'bottom': False}
# Set up the model
grid_1, evolvers = initialise_a_model(dims, model_param_vals, boundaries, deposition=False)

# Set other parameters
runtime_ = df[0]['runtime_kyr']
dt_ = 10.  # Courant timestep [kyr]
# Set threshold for defining steady-state
ss_threshold = 1e-8  # m/m (%)
# Set uplift rate
uplift_rate = 0.2  # m/kyr

# Put forward variables into a dictionary
parameters = {'timestep': dt_,
              'runtime': runtime_,
              'uplift_rate': uplift_rate,
              'steady_state_threshold': ss_threshold}
ic_parameters = {'seed': seed,
                 'amplitude': '1.0m',
                 'colour': 'white'}

# Define forward model function
def evolve_block_to_steady_state_feednoise(mg, initial_condition, evolve_dict,
                                 forward_params, feed_noise, store_all=False):
    """
    Evolve a landscape forwards in time with a user-defined uplift history
    to produce a block uplifted domain. The initial noise is fixed such that it is added at t=0,
    and noise is fed in at every timestep.

    Parameters:
    --------------------------
    mg: Landlab RasterModelGrid
    initial_condition: nd.array of initial topography
    evolve_dict: Dictionary of landscape evolution processes (e.g. sinkfilling, FastScape, etc.)
    forward_params: Dictionary of forward model parameters (e.g. uplift rate, runtime, timestep)
    feed_noise: nd.array of elevations to be added at each timestep.
                Must be of shape n_steps * mg.core_nodes
    store_all: Boolean. If true, return all topographies (i.e. at each timestep).
                        If false, return the final topography only.

    Returns:
    ---------------------------
    topo: nd.array of final topography
    df: pandas DataFrame of various steady-state metrics
    """

    # Assign forward model parameters to variables
    dt = forward_params['timestep']
    runtime = forward_params['runtime']
    uplift_rate_ = forward_params['uplift_rate']

    # Total number of model timesteps
    n_steps = (runtime + dt) // dt

    # Assign topo field to a variable
    z = mg.at_node['topographic__elevation']
    z[mg.core_nodes] = 0

    # Add noise to all nodes
    z[mg.nodes] += initial_condition
    # Then explicitly reset edges nodes to zero elevation
    # Explicitly set boundaries to zero elevation
    for edge in ("left", "top", "bottom", "right"):
        z[mg.nodes_at_edge(edge)] = 0
    evolve_dict['sinkfiller'].run_one_step()

    # Create array to store topo at each timestep
    topos = np.zeros((int(n_steps), mg.shape[0], mg.shape[1]))
    # Create empty lists to store steady-state info
    maxdiff_loc = []
    maxdiff_relief = []
    maxdiff_perc = []
    diff_max = []
    time = []
    diff_mean = []

    # # Print header for further print statements
    # print("Timestep number \t max elev % change \t Model % completion")

    for step in np.arange(n_steps):
        # Copy previous elevation field
        z_old = z.copy()
        # Add uplift
        z[mg.core_nodes] += uplift_rate_ * dt
        # Add extra noise
        z[mg.core_nodes] += feed_noise[int(step)]
        # Flow-route
        evolve_dict['flowrouter'].run_one_step()
        # Erode
        evolve_dict['eroder'].run_one_step(dt)
        # Diffuse
        evolve_dict['diffuser'].run_one_step(dt)
        # Copy the elevations
        topos[int(step)] = np.copy(z).reshape(mg.shape)

        # Assess steady state
        zdiff = np.abs(z[mg.core_nodes] - z_old[mg.core_nodes])
        percent_change = zdiff / z[mg.core_nodes]
        # Record states
        diff_mean.append(
                         np.abs(np.mean(z[mg.core_nodes]) - np.mean(z_old[mg.core_nodes]))
                         )  # change in mean
        diff_max.append(
                        np.abs(np.max(z[mg.core_nodes]) - np.max(z_old[mg.core_nodes]))
                        )  # change in the maximum
        maxdiff_perc.append(np.max(percent_change))  # largest cell-cell % diff in z
        maxdiff_loc.append(np.max(zdiff))  # largest cell-cell diff in z
        maxdiff_relief.append(
                              np.abs(np.max(z[mg.core_nodes]) - np.max(z_old[mg.core_nodes]))
                              )  # largest relief diff
        time.append(dt * int(step))

    df = pd.DataFrame(
            {
                "time [kyr]": time,
                "maxdiff": diff_max,
                "maxdiff_relief": maxdiff_relief,
                "diff_mean": diff_mean,
                "diff_max_loc": maxdiff_loc,
                "maxdiff_perc": maxdiff_perc
            }
        )

    if store_all:
        return topos, df
    return topos[-1], df

# Create an array of noise to feed in by stacking the start topography
def generate_random_coloured_noise(width, height, exponent, max_amplitude, seed_=1):
    """
    Produce a 2D map of coloured noise from an initially random uniform distribution.

    Parameters:
    ------------
    width: int
           Domain width (x dimension)
    height: int
            Domain height (y dimension)
    exponent: float
              Power law exponent by which to scale the frequencies.
              -2 corresponds to red noise
              -1 corresponds to pink noise
              0 corresponds to white noise
              1 corresponds to blue noise
    max_amplitude: int or float
                   Value of maximum amplitude for the noise, from 0 <= noise <= max_amplitude
    seed: int, optional
          Value of the random seed to use. Using the same seed will produced
          repeatable results. Default value is 1

    Returns:
    -----------
    noise: Two dimension nd.array of noise
    """
    # Set random seed
    rng_ = np.random.RandomState(seed_)

    # Create initial white noise array, scaled 0-1 m
    whitenoise = rng_.uniform(0, 1, (height, width))

    # Take Fourier transform and shift to ensure zero central frequency
    white_ft = np.fft.fftshift(np.fft.fft2(whitenoise))

    # Generate a frequency matrix
    _x, _y = np.mgrid[0:white_ft.shape[0], 0:white_ft.shape[1]]  # Generate array of index positions

    # Calculate frequencies by taking hypotenuse.
    # Corresponds to distance from centre of fftshifted fourier space.
    # Larger distance = higher freq.
    f = np.hypot(_x - white_ft.shape[0]/2, _y - white_ft.shape[1]/2)

    # Calculate coloured noise, catch warnings for divide by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        noise_ft = white_ft * np.sqrt(f**exponent)  # still in frequency domain with centered freqs.
        noise_ft = np.nan_to_num(noise_ft, nan=0, posinf=0, neginf=0)  # remove zeros
    noise = np.fft.ifft2(np.fft.ifftshift(noise_ft)).real  # Back to spatial domain, real parts only
    noise += np.abs(np.min(noise))  # Normalise between 0 and 1 by adding min value...
    noise /= np.max(noise)  # ...and dividing by max

    # Scale by max amplitude
    noise *= max_amplitude

    return noise

# Get number of model timesteps
model_steps = int((runtime_ + dt_) // dt_)
# Set exponent to define colour of non-quenched noise
if colour == 'white':
    EXPONENT = 0
elif colour == 'blue':
    EXPONENT = 1
elif colour == 'red':
    EXPONENT = -2
else:
    print("Error - colour is not equal to red/white/blue. \
          Exponent will be set to 0 to feed in white noise.")
    EXPONENT = 0

# Create array of random seeds, using the IC seed value to generate them
# This should ensure repeatability, as all randomness derived from IC seed.
print("### Generating spatio-temporal noise ###")
rng = np.random.RandomState(int(seed))
seeds = rng.randint(1, high=1e9, size=model_steps)
nonquench_noises = []
for ss in seeds:
    random_noise = generate_random_coloured_noise(nx, ny, EXPONENT, amplitude, seed_=ss)
    nonquench_noises.append(random_noise[1:99, 1:99].flatten())

# Run forward model
print("### Running forward model ###")
landscapes, ss_data = evolve_block_to_steady_state_feednoise(grid_1, start_topo, evolvers,
                                                   parameters, nonquench_noises, store_all=True)
# Run geomorphic components
print("### Extracting geomorphic data from final landscape ###")
# Make directory for individual seed
geomorphic_data_savedir = f"{data_savedir}/geomorphic_data/seed_{seed}"
if not os.path.exists(geomorphic_data_savedir):
    os.makedirs(geomorphic_data_savedir)

min_drainage_area = 4e6  # sq m
ref_concavity = 0.5
cp = ChannelProfiler(grid_1,
                        number_of_watersheds=4, 
                        minimum_channel_threshold=min_drainage_area,
                        main_channel_only=False)
cp.run_one_step()
# Chi
cf = ChiFinder(grid_1,
               min_drainage_area = min_drainage_area,
               reference_concavity = ref_concavity,
               use_true_dx = False,
               clobber = True)
cf.calculate_chi()
chi_data = extract_chi_profile_data(grid_1, cp, save=True, savedir=geomorphic_data_savedir)
# Hack exponent
hc = HackCalculator(grid_1,
                number_of_watersheds=4,
                main_channel_only=True,
                save_full_df=True)
hc.calculate_hack_parameters()
summary_data, full_data = extract_hack_data(hc, geomorphic_data_savedir)

# Save results
print("### Saving forward model results ###")
# Save grid
save_grid(grid_1, f"{data_savedir}/grids/grid_{seed}.grid", clobber=True)
# Save ChannelProfiler
cp_fname = f"{data_savedir}/profilers/cp_seed_{seed}.pkl"
with open(cp_fname, 'wb') as file:
    pickle.dump(cp, file, -1)
# np.save(data_savedir+'/topos_' + seed +'.npy', landscapes)
ss_data.to_csv(
    f'{data_savedir}/ss_info/ss_data_{uplift_rate}_{int(runtime_/1e3)}Myr_evolution_{seed}.csv'
    )
# Save forward input file
shutil.copyfile(forward_filepath, data_savedir+'/inputs/forward_model_inputs_' + seed + '.csv')

print("### DONE ###")