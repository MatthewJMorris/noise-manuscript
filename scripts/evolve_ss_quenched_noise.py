"""
Evolve a coloured landscape with added quenched noiseto steady state
 MM Jan 2025
"""
# Load required packages
import os
import shutil
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
data_output_quenched = os.path.join(data_output_ss, dirs["quenched"])

# Read infile to get inputs
infile = np.loadtxt(data_input_ss + 'tmp_infile.txt', delimiter=',', dtype='str', skiprows=1)
ic_filepath = os.path.join(curr_dir, infile[0].strip())
forward_filepath = os.path.join(curr_dir, infile[1].strip())

# Extract IC file information to get the seed and save directory
ic_file = ic_filepath.split("/")[-1].split("_")
seed = ic_file[-1][:-4]  # last part of filename, all but last 4 chars
colour = ic_file[1]
amplitude = ic_file [2]
print(f"### {colour} {amplitude} {seed} ###")
save_filepath = ic_filepath.split("/")[-3:-1]
save_filepath = "/".join(save_filepath)
data_savedir = os.path.join(data_output_quenched, save_filepath)
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

# Set other parameter values
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
    and is also fed in at every subsequent timestep.

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

    # Create array to store topo at every 10th timestep
    topos = np.zeros((int(n_steps/10)+1, mg.shape[0], mg.shape[1]))
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
        # Copy every 10th timestep of elevations
        if step % 10 == 0:
            topos[int(step/10)] = np.copy(z).reshape(mg.shape)

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
model_steps = int((runtime_ + dt_) // dt_)
quench_noise = np.tile(start_topo[1:99, 1:99].flatten(), (model_steps,1))

# Run forward model
print("### Running forward model ###")
landscapes, ss_data = evolve_block_to_steady_state_feednoise(grid_1, start_topo, evolvers,
                                                   parameters, quench_noise, store_all=True)
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

# Done
print("### DONE ###")