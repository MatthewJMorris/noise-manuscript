""" Suite of functions used for geomorphic analysis of topography
    & data extracted from Landlab"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pygmt


def calculate_normalised_hypsometry(topography):
    """ 
    Calculate the normalised hypsometric curve of topography

    Parameters:
    ---------------
    topography: nd.array
                2D array of elevations
    
    Returns:
    ---------------
    area_norm: nd.array
               Array of size topography corresponding to relative area, between 0 and 1.
    hyps_norm: nd.array
               Normalised and sorted elevations
    """

    # Calculate hypsometry
    hypsometry = np.sort(topography.flatten())
    hyps_norm = hypsometry / np.max(hypsometry)
    
    # Calculate area
    area = np.arange(0, topography.size, 1)
    area_norm = area / area[-1]

    return area_norm, hyps_norm


def extract_long_profile_data(mg, profiler, save=False, savedir=None):
    """
    Extract (and optionally save) longitudinal profiles from a LandLab ChannelProfiler object.

    Parameters:
    --------------
    mg: LandLab RasterModelGrid object
    profiler: LandLab ChannelProfiler object
    save: bool, optional
          Determine whether to save the data
    savedir: str, optional
          A directory path for where to save the data

    Returns:
    -------------
    basins: Nested list of tuples
            Each list represents a single basin. Each sublist is a river segment, 
            containing tuples of upstream distance and elevation data.
    """

    # Initialise an empty list in which to store rivers for each basin
    basins = []
    # Loop over each basin, identified by its outlet ID
    for outlet in profiler.data_structure:
        sect = profiler.data_structure[outlet].keys()  # sections of the basin network
        # (Re)set segments to an empty list
        segments = []
        # Store the current length of basins array,
        # required for correct indexing on distance_along_profile
        offset = len(sum(basins, []))
        for idx, segment in enumerate(sect):
            # Get nodes in each segment
            nodes = profiler.data_structure[outlet][segment]["ids"]
            # Convert to grid cell nodes and store
            segments.append(list(zip(profiler.distance_along_profile[idx+offset],
                                     mg.at_node['topographic__elevation'][nodes])))
        # Store each basin network
        basins.append(segments)

    if save:
        if savedir is None:
            print("Save option is True, but directory not provided. Saving in ./tmp/")
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
                # Write each watershed into a separate file
                for outlet_id, basin in zip(profiler.data_structure, basins):
                    with open(f'./tmp/long_profiles_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                        for segment in basin:
                            f.write(f"{str(segment)}\n")
        else:
            # Write each watershed into a separate file
            for outlet_id, basin in zip(profiler.data_structure, basins):
                with open(f'{savedir}/long_profiles_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                    for segment in basin:
                        f.write(f"{str(segment)}\n")

    return basins


def load_long_profile_data(file):
    """
    Load longitudinal profile data from a file, 
    initially saved from a LandLab ChannelProfiler object.

    Parameters:
    ------------
    file: str
          filepath of longitudinal profile data to be loaded

    Returns:
    -----------
    basin: Nested list of tuples.
           Each list is a channel segment containing (upstream distance, elevation) information
    """
    # Initialise empty list
    basin = []
    # Read the file
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove the newline and convert from string representation to an actual list
            line = eval(line.rstrip("\n"))
            basin.append(line)
    return basin


def extract_and_plot_long_profile_map(mg, profiler, landscape_properties,
                                      save=False, savedir=None):
    """ 
    Extract longitudinal profiles from Landlab ChannelProfiler
    and plot in map view

    Parameters:
    ----------------
    mg: Landlab RasterModelGrid
    profiler: LandLab ChannelProfiler
    landscape_properties: dict
                          Relevant information about the landscape (e.g. age, seed, colour)
    save: bool, optional
          Determine whether to save the figure
    savedir: str, optional
             A directory path for where to save the figure
    """

    # Extract elevations from the grid
    topo_field = xr.DataArray(mg.at_node['topographic__elevation'].reshape(mg.shape),
                          coords={"x": (mg.x_of_node.reshape(mg.shape)/mg.dx)[0],
                                  "y": (mg.y_of_node.reshape(mg.shape)/mg.dy)[:,0]}
                                  )
    
    # Extract info from landscape_properties dictionary
    seed = landscape_properties['seed']
 
    # Extract the drainage network stored in the ChannelProfiler
    # First initialise an empty list in which to store rivers for each basin 
    basins = []
    # Loop over each basin, identified by its outlet ID
    for ii, outlet in enumerate(profiler.data_structure):
        sect = profiler.data_structure[outlet].keys()  # sections of the basin network
        # (Re)set segments to an empty list
        segments = []
        for segment in sect:
            # Get nodes in each segment
            nodes = profiler.data_structure[outlet][segment]["ids"]
            # Convert to grid cell nodes and store
            segments.append(list(zip(mg.x_of_node[nodes], mg.y_of_node[nodes])))
        # Store each basin network
        basins.append(segments)

    # Move on to plotting
    fig = pygmt.Figure()
    with pygmt.config(FONT="8p,Helvetica,black", MAP_FRAME_PEN="1p"):
        # First plot the underlying topography
        pygmt.grd2cpt(topo_field, cmap="terra", continuous=True, reverse=False)
        fig.grdimage(topo_field, cmap=True,
                     frame=["WrSt", "a20f10+lDistance [km]"], projection="X8c")
        fig.colorbar(position="JTC+w8c+o0c/0.3c+h", frame=["a200f100", "x+lElevation [m]"])
        # Add the drainage network on top
        # Get colors from a cpt
        cb_annots = [str(key) for key in profiler.data_structure]
        pygmt.makecpt(cmap="inferno",
                    series=[1,len(cb_annots),1],
                    color_model="+c" + ",".join(cb_annots))
        for ii, basin in enumerate(basins):
            colour = ii+1
            for seg in basin:
                fig.plot(x=[i[0]/1000 for i in seg], y=[i[1]/1000 for i in seg],
                         zvalue=colour, pen="1p,+z", cmap=True)
    fig.show()

    if save:
        if savedir is None:
            print("Save option is True, but directory not provided. Saving in figs/tmp/")
            if not os.path.exists('figs/tmp'):
                os.makedirs('figs/tmp')
            fig.savefig('figs/tmp/drainage_map.png', dpi=400)
        else:
            fig.savefig(f'{savedir}/{seed}_drainage_map.png', dpi=400)


def extract_chi_profile_data(mg, profiler, save=False, savedir=None):
    """
    Extract (and optionally save) chi profiles from a LandLab ChannelProfiler object.

    Parameters:
    --------------
    mg: LandLab RasterModelGrid object
    profiler: LandLab ChannelProfiler object
    save: bool, optional
          Determine whether to save the data
    savedir: str, optional
          A directory path for where to save the data

    Returns:
    -------------
    chi_profiles: Nested list of tuples
                  Each list represents a single basin. Each sublist is a river segment, 
                  containing tuples of chi metric and elevation data.
    """

    chi_profiles = []
    for ii, outlet_id in enumerate(profiler.data_structure):
        # (Re)set segments to an empty list
        segments = []
        for jj, segment_id in enumerate(profiler.data_structure[outlet_id]):
            segment = profiler.data_structure[outlet_id][segment_id]
            profile_ids = segment["ids"]
            # Convert to grid cell nodes and store
            segments.append(list(zip(mg.at_node['channel__chi_index'][profile_ids],
                                     mg.at_node['topographic__elevation'][profile_ids])))
        # Store each basin network
        chi_profiles.append(segments)

        if save:
            if savedir is None:
                print("Save option is True, but directory not provided. Saving in ./tmp/")
                if not os.path.exists('./tmp'):
                    os.makedirs('./tmp')
                    # Write each watershed into a separate file
                    for outlet_id, basin in zip(profiler.data_structure, chi_profiles):
                        with open(f'./tmp/chi_profiles_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                            for segment in basin:
                                f.write(f"{str(segment)}\n")
            else:
                # Write each watershed into a separate file
                for outlet_id, basin in zip(profiler.data_structure, chi_profiles):
                    with open(f'{savedir}/chi_profiles_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                        for segment in basin:
                            f.write(f"{str(segment)}\n")

    return chi_profiles


def load_chi_profile_data(file):
    """
    Load chi profile data from a file, 
    initially saved from a LandLab ChannelProfiler/ChiFinder object.

    Parameters:
    ------------
    file: str
          filepath of chi profile data to be loaded

    Returns:
    -----------
    chi_profiles: Nested list of tuples.
                  Each list is a channel segment containing (chi index, elevation) information
    """

    chi_profiles = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            line = eval(line.rstrip("\n"))
            chi_profiles.append(line)

    return chi_profiles


def extract_ksn_profiles(mg, profiler, save=False, savedir=None):
    """
    Extract (and optionally save) normalised channel steepnes
    from a LandLab ChannelProfiler object.

    Parameters:
    --------------
    mg: LandLab RasterModelGrid object
    profiler: LandLab ChannelProfiler object
    save: bool, optional
          Determine whether to save the data
    savedir: str, optional
          A directory path for where to save the data

    Returns:
    -------------
    ksn_profiles: Nested list of tuples
                  Each list represents a single basin. Each sublist is a river segment, 
                  containing tuples of upstream distance and channel steepness.
    """

    ksn_profiles = []
    for ii, outlet_id in enumerate(profiler.data_structure):
        # (Re)set segments to an empty list
        segments = []
        for jj, segment_id in enumerate(profiler.data_structure[outlet_id]):
            segment = profiler.data_structure[outlet_id][segment_id]
            profile_ids = segment["ids"]
            distance_upstream = segment["distances"]
            # Convert to grid cell nodes and store
            segments.append(list(zip(distance_upstream,
                                     mg.at_node['channel__steepness_index'][profile_ids])))
        # Store each basin network
        ksn_profiles.append(segments)

        if save:
            if savedir is None:
                print("Save option is True, but directory not provided. Saving in ./tmp/")
                if not os.path.exists('./tmp'):
                    os.makedirs('./tmp')
                    # Write each watershed into a separate file
                    for outlet_id, basin in zip(profiler.data_structure, ksn_profiles):
                        with open(f'./tmp/ksn_profiles_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                            for segment in basin:
                                f.write(f"{str(segment)}\n")
            else:
                # Write each watershed into a separate file
                for outlet_id, basin in zip(profiler.data_structure, ksn_profiles):
                    with open(f'{savedir}/ksn_profiles_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                        for segment in basin:
                            f.write(f"{str(segment)}\n")

    return ksn_profiles


def load_ksn_profile_data(file):
    """
    Load ksn profile data from a file, 
    initially saved from a LandLab ChannelProfiler/SteepnessFinder object.

    Parameters:
    ------------
    file: str
          filepath of ksn profile data to be loaded

    Returns:
    -----------
    ksn_profiles: Nested list of tuples
                  Each sublist is a channel segment containing (upstream distance, steepness)
    """

    ksn_profiles = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            line = eval(line.rstrip("\n"))
            ksn_profiles.append(line)

    return ksn_profiles


def extract_slope_area_data(mg, profiler, save=False, savedir=None):
    """
    Extract (and optionally save) slope-area
    from a LandLab RasterModelGrid/ChannelProfiler object.

    Parameters:
    --------------
    mg: LandLab RasterModelGrid object
    profiler: LandLab ChannelProfiler object
    save: bool, optional
          Determine whether to save the data
    savedir: str, optional
          A directory path for where to save the data

    Returns:
    -------------
    slope_area: Nested list of tuples
                  Each list represents a single basin. Each sublist is a river segment, 
                  containing tuples of drainage area and topographic slope.
    """

    slope_area = []
    for ii, outlet_id in enumerate(profiler.data_structure):
        # (Re)set segments to an empty list
        segments = []
        for jj, segment_id in enumerate(profiler.data_structure[outlet_id]):
            segment = profiler.data_structure[outlet_id][segment_id]
            profile_ids = segment["ids"]
            # Convert to grid cell nodes and store
            segments.append(list(zip(mg.at_node['drainage_area'][profile_ids],
                                     mg.at_node['topographic__steepest_slope'][profile_ids])))
        # Store each basin network
        slope_area.append(segments)

        if save:
            if savedir is None:
                print("Save option is True, but directory not provided. Saving in ./tmp/")
                if not os.path.exists('./tmp'):
                    os.makedirs('./tmp')
                    # Write each watershed into a separate file
                    for outlet_id, basin in zip(profiler.data_structure, slope_area):
                        with open(f'./tmp/chi_profiles{outlet_id}.txt', 'w', encoding='utf-8') as f:
                            for segment in basin:
                                f.write(f"{str(segment)}\n")
            else:
                # Write each watershed into a separate file
                for outlet_id, basin in zip(profiler.data_structure, slope_area):
                    with open(f'{savedir}/slope_area_{outlet_id}.txt', 'w', encoding='utf-8') as f:
                        for segment in basin:
                            f.write(f"{str(segment)}\n")

    return slope_area


def load_slope_area_data(file):
    """
    Load slope_area data from a file, 
    initially saved from a LandLab RasterModelGrid/ChannelProfiler object.

    Parameters:
    ------------
    file: str
          filepath of slope-area data to be loaded

    Returns:
    -----------
    slope_area: list
           Nested list of tuples. Each list is a channel segment containing (drainage area, slope) information
    """

    slope_area = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            line = eval(line.rstrip("\n"))
            slope_area.append(line)

    return slope_area


def extract_hack_data(hack_calculator, savedir):
    """
    Save the DataFrames produced by LandLab HackCalculator component

    Parameters:
    ------------
    hack_calculator: LandLab HackCalculator object
    savedir: str
             Filepath at which to save the DataFrames
             
    Returns:
    -----------
    summary_data: DataFrame of Hack exponent per basin
    full_data: DataFrame of all area-length data
    """

    # Assign to variables
    summary_data = hack_calculator.hack_coefficient_dataframe
    full_data = hack_calculator.full_hack_dataframe

    # Save
    summary_data.to_csv(f"{savedir}/hack_exponent_data.csv", index=True)
    full_data.to_csv(f"{savedir}/hack_full_data.csv", index=True)

    return summary_data, full_data


def load_hack_data(dirpath):
    """
    Load Hack exponent csvs as DataFrames from a directory path

    Parameters:
    -------------
    dirpath: str
             Filepath from which to load Hack data
    
    Returns:
    ------------
    hack_exponent_data: DataFrame of summary data
    hack_full_data: DataFrame of full data
    """

    hack_exponent_data = pd.read_csv(f"{dirpath}/hack_exponent_data.csv", index_col='basin_outlet_id')
    hack_full_data = pd.read_csv(f"{dirpath}/hack_full_data.csv", index_col='node_id')

    return hack_exponent_data, hack_full_data
