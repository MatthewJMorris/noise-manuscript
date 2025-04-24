#!/usr/bin/env python3

import numpy as np


"""
A series of useful landscape evolution functions
"""


def initialise_forward_model(input_file, nx=100, ny=100, dx=1000):
    """
    Loads forward model input parameters and initialises components within Landlab.

    Parameters:
    ------------
    input_file: csv file of forward model parameter values
    nx: int, optional
        Number of x grid cells. Default is 100
    ny: int, optional
        Number of y grid cells. Default is 100
    dx: float, optional
        x-step (metres). Assume y step is the same. Default is 1000

    Returns:
    -----------
    mg: LandLab RasterModelGrid object
    z: Topographic elevation data field associated with mg
    evolvers_: Dictionary of model components (e.g. FlowAccumulator, LinearDiffuser, etc.)
    params_: Dictionary of other relevant model parameters (e.g. runtime)
    """
    from landlab import RasterModelGrid
    from landlab.components import FlowAccumulator, FastscapeEroder, SinkFillerBarnes, LinearDiffuser
    import pandas as pd
    df = pd.read_csv(input_file,
                     dtype={'runtime_kyr': float, 'n_sp': float, 'm_sp': float, 'K_sp': float,
                            'D_diff': float}, skipinitialspace=True).to_dict(orient='records')

    # Initialise parameters for the forward model
    mg = RasterModelGrid((nx, ny), dx, xy_axis_units='m')
    z = mg.add_zeros('topographic__elevation', at='node')
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    n_sp = df[0]['n_sp']
    m_sp = df[0]['m_sp']
    K_sp = df[0]['K_sp']
    D_diff = df[0]['D_diff']
    fr = FlowAccumulator(mg, flow_director='D8')
    diff = LinearDiffuser(mg, linear_diffusivity=D_diff, deposit=False)
    sp = FastscapeEroder(mg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp)
    sb = SinkFillerBarnes(mg, method='D8')
    evolvers_ = {'flowrouter': fr, 'diffuser': diff, 'eroder': sp, 'sinkfiller': sb}
    runtime_ = df[0]['runtime_kyr']
    params_ = {'runtime': runtime_}

    return mg, z, evolvers_, params_


def initialise_a_model(dimensions, component_values, boundary_conditions, deposition):
    """
    A function to rapidly set up a model for experiments within this notebook

    Parameters:
    ------------
    dimensions: dict
                Contains number of y cells, number of x cells, cell spacing
    component_values: dict
                        Contains values for the forward model components
    boundary_conditions: dict of bool
                            True (closed) or False (open) for [right, top, left, bottom] boundaries
    deposition: bool
                If True then deposition is turned on within the LinearDiffuser component.
                Turned off if False

    Returns:
    -----------
    mg: Landlab RasterModelGrid object
    evolvers_: Dictionary of components
    """
    from landlab import RasterModelGrid
    from landlab.components import FlowAccumulator, FastscapeEroder, SinkFillerBarnes, LinearDiffuser
    
    mg = RasterModelGrid((dimensions['ny'], dimensions['nx']), dimensions['dx'], xy_axis_units='m')
    mg.add_zeros('topographic__elevation', at='node')
    mg.set_closed_boundaries_at_grid_edges(boundary_conditions['right'],
                                           boundary_conditions['top'],
                                           boundary_conditions['left'],
                                           boundary_conditions['bottom']) 

    # Instantiate components
    fr = FlowAccumulator(mg, flow_director='D8')
    sp = FastscapeEroder(mg,
                         K_sp=component_values['K_sp'],
                         m_sp=component_values['m_sp'],
                         n_sp=component_values['n_sp'])
    sb = SinkFillerBarnes(mg, method='D8')
    if deposition:
        diff = LinearDiffuser(mg, linear_diffusivity=component_values['D_diff'], deposit=True)
    else:
        diff = LinearDiffuser(mg, linear_diffusivity=component_values['D_diff'], deposit=False)

    
    evolvers_ = {'flowrouter': fr, 'diffuser': diff, 'eroder': sp, 'sinkfiller': sb}

    return mg, evolvers_


###########################
# Forward model functions #
###########################
def gaussian_dome_to_grid(mg, n, scale, b=2):
    """
    Creates a Gaussian dome and applies 
    to a LandLab RasterModelGrid.
    Return z elevations at x, y, coordinates.
    """
    x, y = np.meshgrid(np.linspace(-b, b, n), np.linspace(-b, b, n))
    z = np.exp(-x * x - y * y) * scale

    # Fit into the larger grid dimensions
    dy = mg.shape[1]
    # Add zeros to x axis
    z = np.append(np.insert(z, 0, np.zeros((int((dy - n) / 2), n)), axis=0), np.zeros((int((dy - n) / 2), n)), axis=0)
    # Add zeros to y axis
    z = np.append(np.insert(z, 0, np.zeros((int((dy - n) / 2), dy)), axis=1), np.zeros((dy, int((dy - n) / 2))), axis=1)

    return x, y, z


def gaussian_dome_to_grid_2(x_centre, y_centre, sigma_x, sigma_y, x_size, y_size, mg, scale, theta=0):
    """
    Creates a Gaussian dome in a box and fits the box to grid
    x_centre: Relative x position of Gaussian centre. Bounds 0 to 1
    y_centre: Relative y position of Gaussian centre. Bounds 0 to 1
    theta: angle of rotation for Gaussian. Bounds 0 to 1
    sigma_x: standard deviation in x-axis. Spread in x of Gaussian
    sigma_y: standard deviation in y-axis. Spread in y of Gaussian
    x_size: box diameter in x dimension. Cell units
    y_size: box diameter in y dimension. Cell units
    mg: Landlab model grid
    scale: Scaling factor for peak elevation
    Adapted from https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    """
    # Check that x and y size do not exceed size of model grid
    assert x_size <= mg.number_of_node_columns, "Box x_size exceeds grid x dimension"
    assert y_size <= mg.number_of_node_rows, "Box y_size exceeds grid y dimension"

    x_bound = (0.5 * x_size) / mg.number_of_node_columns
    y_bound = (0.5 * y_size) / mg.number_of_node_rows

    theta = 2 * np.pi * theta / 360
    x = np.arange(0, x_size, 1, float)
    y = np.arange(0, y_size, 1, float)
    y = y[:, np.newaxis]
    x0 = 0.5 * x_size  # x centre of Gaussian manually defined at centre of box
    y0 = 0.5 * y_size  # y centre of Gaussian manually defined at centre of box
    sx = sigma_x
    sy = sigma_y

    # rotation
    a = np.cos(theta) * x - np.sin(theta) * y
    b = np.sin(theta) * x + np.cos(theta) * y
    a0 = np.cos(theta) * x0 - np.sin(theta) * y0
    b0 = np.sin(theta) * x0 + np.cos(theta) * y0

    z = np.exp(-(((a - a0) ** 2) / (2 * (sx ** 2)) + ((b - b0) ** 2) / (2 * (sy ** 2))))
    if x_size == mg.number_of_node_rows and y_size == mg.number_of_node_columns:
        return z * scale
    else:  # Fit into larger grid dimensions
        dx = mg.shape[1]
        dy = mg.shape[0]
        # x_cells_to_add = int(dx - x_size)
        # y_cells_to_add = int(dy - y_size)
        xpos_insert = x_centre - x_bound  # Relative position to insert on x-axis
        # if (dx - x_size)/dx < xpos_insert <= (dx - x_size + 0.6)/dx:
        #     xpos_insert = (dx - x_size + 0.5) / dx
        xpos_append = round((1. - x_centre) - x_bound, 14)  # Relative position to append on x-axis
        if -1e-2 <= xpos_append < 0:
            xpos_append = 0.
        ypos_insert = y_centre - y_bound
        # if (dy - y_size)/dy < ypos_insert <= (dy - y_size + 0.6)/dy:
        #     ypos_insert = (dy - y_size + 0.5) / dy
        ypos_append = round((1. - y_centre) - y_bound, 14)  # bit of a dirty fix to overcome floating point errors < 0
        if -1e-2 <= ypos_append < 0:
            ypos_append = 0.
        # print("HERE 1", z.shape)
        if not x_bound <= x_centre <= (1 - x_bound) or not y_bound <= y_centre <= (1 - y_bound):
            print("HERE 2", z.shape)
            print(ypos_insert, ypos_append, xpos_insert, xpos_append)
            print(y_centre, y_bound)
            if ypos_insert < 0:
                z = z[round(abs(ypos_insert * dy)):]
                z = np.append(z, np.zeros((round((dy * ypos_append)), x_size)), axis=0)
            elif ypos_append < 0:
                z = np.insert(z, 0, np.zeros((round(dy * ypos_insert), x_size)), axis=0)
                z = z[:round(dy * ypos_append)]
            else:
                z = np.append(np.insert(z, 0, np.zeros((round(dy * ypos_insert), x_size)), axis=0),
                              np.zeros((round((dy * ypos_append)), x_size)), axis=0)
            #print("HERE 3", z.shape)
            if xpos_insert < 0:
                z = z[:, round(abs(xpos_insert * dx)):]
                z = np.append(z, np.zeros((dy, round(dx * xpos_append))), axis=1)
            elif xpos_append < 0:
                z = np.insert(z, 0, np.zeros((round(dx * xpos_insert), dy)), axis=1)
                z = z[:, :round(xpos_append * dx)]
            else:
                #print("HERE 4", z.shape)
                z = np.append(np.insert(z, 0, np.zeros((round(dx * xpos_insert), dy)), axis=1),
                              np.zeros((dy, round(dx * xpos_append))), axis=1)

            # Final check for dimensions
            if z.shape[1] > dx:
                z = z[:, 0:dx]  # If too many x inserted, slice down to size

            return z * scale

        else:
            # Add zeros to y axis
            z = np.append(np.insert(z, 0, np.zeros((round(dy * ypos_insert), x_size)), axis=0),
                          np.zeros((round((dy * ypos_append)), x_size)), axis=0)
            # Add zeros to x axis
            z = np.append(np.insert(z, 0, np.zeros((round(dx * xpos_insert), dy)), axis=1),
                          np.zeros((dy, round(dx * xpos_append))), axis=1)
            return z * scale



###########################
# Miscellaneous functions #
###########################
def calculate_courant(A, dx=1e3, v=1e-6, m=0.5):
    """
    Calculates a timestep which satisfies the courant condition
    based on a user inputted area
    returns dt, yrs
    """
    dt = dx / (v * A ** m)
    return round(dt, 0)


def calculate_response_time(grid_, K, m, node_cell_area, boundary_nodes=None):
    """
    Calculates the landscape response time, or Gilbert time, for a landscape.
    I.e. the time for a knickpoint to propagate through a landscape from base level upstream
    As per Eq 6 in Roberts et al. 2012 (Colorado).
    t_G = sum(dx /k*A^m), where k = erosivity, x = point on the river profile,
    A = drainage area, m = constant.
    Adapted from landlab Flow Accumulation function find_drainage_area_and_discharge.

    Parameters:
    grid_: Landlab model grid
    K: Erodibility [m^{1-2m}yr^-1]. float.
    m: Erosional exponent. float
    node_cell_area: float or ndarray. Cell surface area for each node.
    boundary_nodes: list, optional. Array of boundary nodes to have drainage area and response time set to zero

    Returns landscape response time in units of time

    """
    # Calculate upstream flow distances using landlab function
    from landlab.utils.flow__distance import calculate_flow__distance
    # flow_dist = calculate_flow__distance(grid_)  # represents dx

    # Define per cell area
    area = node_cell_area

    # Downstream to upstream array of node ID's
    s = grid_.at_node["flow__upstream_node_order"]
    # Receiver ID's for each node
    r = grid_.at_node["flow__receiver_node"]

    # Number of points
    n = len(s)

    # Initial values for drainage area and response time
    drainage_area = np.zeros(n, dtype=int) + area  # Start with a base drainage area of per cell area
    # response_time = np.zeros(n, dtype=int)

    # Optionally zero out drainage area at boundary nodes
    if boundary_nodes is not None:
        drainage_area[boundary_nodes] = 0

    # function to calculate upstream drainage area
    for ii in range(n - 1, -1, -1):  # Iterate through index backwards (upstream to downstream)
        donor = s[ii]  # For each node, which node flows into it
        recvr = r[donor]  # For each node, which node does it flow to
        if donor != recvr:
            drainage_area[recvr] += drainage_area[donor]

    # Cell to cell flow distances
    cell_dists = grid_.length_of_d8[grid_.at_node["flow__link_to_receiver_node"]]

    # Function to be integrated upstream
    integrand = cell_dists / (drainage_area ** m)

    # Integrate upstream for each channel
    integral = np.zeros(n) + integrand
    for ii in range(0, n , 1):
        donor = s[ii]  # donor represents the upstream node
        recvr = r[donor]  # receiver represents the downstream node
        if donor != recvr:
            integral[donor] += integral[recvr]

    # Multiply by 1/v
    response_time = (1/K) * integral

    # Calculate response time
    # response_time = flow_dist / (drainage_area ** m) * (1 / K)
    return response_time
