"""
Generate two-dimensional random noise maps 
with power spectral densities of red/white/blue
M Morris September 2024
"""
from configparser import ConfigParser
import os
import warnings
import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from scripts.fourier_functions import (rapsd, calculate_2d_ft,
                                       calculate_averages_in_bins,
                                       collapse_2d_psd_to_1d)

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
          repeatable results. Default value is 1.

    Returns:
    -----------
    noise: Two dimension nd.array of noise
    """
    
    # Set random seed
    np.random.seed(seed_)

    # Create initial white noise array, scaled 0-1 m
    whitenoise = np.random.uniform(0, 1, (height, width))

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


# Load config file and extract directories
config_obj = ConfigParser()
config_obj.read('config.ini')
dirs = config_obj["directories"]
data_input_main = dirs["data_input"]
data_input_ic = os.path.join(data_input_main, dirs["input_initial_conditions"])
# Concatenate to get the output directory
data_savedir = os.path.join(data_input_ic, dirs['random'])

# Get input parameters from file
input_data = pd.read_csv(data_input_ic + 'tmp.csv',
                        dtype={'noise_type': str, 'width': int, 'height': int,
                               'max_amplitude': float, 'seed': int},
                                skipinitialspace=True).to_dict(orient='records')

xdim = input_data[0]['width']
ydim = input_data[0]['height']
max_amp = input_data[0]['max_amplitude']
seed = input_data[0]['seed']

# Generate noise
print("### Generating random noise ### ")
red = generate_random_coloured_noise(xdim, ydim, -2, max_amp, seed_=seed)
white = generate_random_coloured_noise(xdim, ydim, 0, max_amp, seed_=seed)
blue = generate_random_coloured_noise(xdim, ydim, 1, max_amp, seed_=seed)

# Create the full path to save noise and plots
red_savedir = data_savedir + 'red_' + str(max_amp) + 'm'
white_savedir = data_savedir + 'white_' + str(max_amp) + 'm'
blue_savedir = data_savedir + 'blue_' + str(max_amp) + 'm'
for dir_path in [red_savedir, white_savedir, blue_savedir]:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path + '/figs', exist_ok=True)
        os.makedirs(dir_path + '/inputs', exist_ok=True)

# Save input file as csv
print("### Saving input file ###")
df = pd.DataFrame([input_data[0]])  # convert back to dataframe
df.to_csv(f'{red_savedir}/inputs/input_data_{seed}.csv', index=False, header=True)
df.to_csv(f'{white_savedir}/inputs/input_data_{seed}.csv', index=False, header=True)
df.to_csv(f'{blue_savedir}/inputs/input_data_{seed}.csv', index=False, header=True)

# Save noise
print("### Saving noise ###")
np.save(f'{red_savedir}/random_red_{max_amp}m_{seed}.npy', red)
np.save(f'{white_savedir}/random_white_{max_amp}m_{seed}.npy', white)
np.save(f'{blue_savedir}/random_blue_{max_amp}m_{seed}.npy', blue)

# Generate data if we wish to produce plots
# Verify that power is as expected by calculating radial power
# and the 2D power spectrum collapsed into 1D
red_radial, freq_r = rapsd(red, fft_method=np.fft, return_freq=True)
white_radial, freq_w = rapsd(white, fft_method=np.fft, return_freq=True)
blue_radial, freq_b = rapsd(blue, fft_method=np.fft, return_freq=True)
# Calculate 2D power spectrum
red_psd = np.abs(calculate_2d_ft(red))**2
white_psd = np.abs(calculate_2d_ft(white))**2
blue_psd = np.abs(calculate_2d_ft(blue))**2
# Normalise by size
red_psd /= red_psd.size**2
white_psd /= white_psd.size**2
blue_psd /= blue_psd.size**2

# Collapse to 1D
red_psd_collapsed = collapse_2d_psd_to_1d(red_psd)
white_psd_collapsed = collapse_2d_psd_to_1d(white_psd)
blue_psd_collapsed = collapse_2d_psd_to_1d(blue_psd)

# Get shifted frequencies.
# Will be the same for each as the dimensions are the same
x_freqs = np.fft.fftshift(np.fft.fftfreq(red_psd.shape[1]))
y_freqs = np.fft.fftshift(np.fft.fftfreq(red_psd.shape[0]))

# Create 25 equally spaced bins and get bin averages
bins_ = np.logspace(-2.2, -0.3, 25)
white_bin_means = calculate_averages_in_bins(white_psd_collapsed, bins_)
blue_bin_means = calculate_averages_in_bins(blue_psd_collapsed, bins_)
red_bin_means = calculate_averages_in_bins(red_psd_collapsed, bins_)

# Plotting function - pygmt
def plot_noise_and_power(noise, psd_2D, psd_collapsed, radial_power, radial_freq,
                         bin_data, bin_mean_data, settings_dict, save=False, savedir=None):
    """
    Creates a three panel plot of (a) the noise map, (b) the 2D power spectrum,
    and (c) various representations of power in 1D

    Parameters:
    -------------------
    noise: xarray DataArray
           2D array of noise
    psd_2D:   xarray DataArray
              power spectrum of the noise array
    psd_collapsed:  nd.array
                    array of radial distance and power for each cell in 2D spectrum
    radial_power:   nd.array
                    Radially averaged power spectrum
    radial_freq:    nd.array
                    Frequencies corresponding to the radially averaged power spectrum
    bin_data:   nd.array
                Logarithmically equally spaced bin values
    bin_mean_data:  nd.array
                    Mean power for each bin
    settings_dict: dictionary
                   Contains seed, colour of noise, and maximum amplitude
    save: bool, optional
          Determine whether to save the figure
    savedir: str, optional
             A directory path for where to save the figure

    """

    # Extract dictionary info into variables
    seed_ = settings_dict['seed']
    colour = settings_dict['colour']
    peak_amp = settings_dict['max_amp']

    # Ensure colour is either red, white, or blue
    colours = ['red', 'white', 'blue']
    assert colour in colours, "colour should be one of 'red', 'white', 'blue'"

    # Get shifted frequencies
    yfreqs = np.fft.fftshift(np.fft.fftfreq(noise.shape[0]))
    xfreqs = np.fft.fftshift(np.fft.fftfreq(noise.shape[1]))

    fig = pygmt.Figure()
    with pygmt.config(FONT_ANNOT_PRIMARY="10p,Helvetica,black",
                      MAP_FRAME_PEN="1p, black",
                      FONT_LABEL="auto"):
        # Plot noisy initial topography
        fig.grdimage(noise, cmap="batlowW", projection="X8c",
                     frame=["WeSt", "xa20f10+lDistance [km]", "ya20f10+lDistance [km]"],
                     region=[0, 99, 0, 99])
        fig.text(text='a', position='TL', justify='TL',
                 offset='0.1c/-0.1c', font="12p", fill='white')
        fig.text(text=rf'Seed = {seed_}, z@-max@- = {peak_amp} m',
                 position='TC', justify='TC', offset='0c/-0.1c', font="12p", fill='white')
        fig.colorbar(position="JTC+w8c+o0c/0.3c+h",
                     frame=[f"a{(peak_amp/4)}f{(peak_amp/8)}+lElevation [m]"])
        # 2D Power spectrum (not normalised)
        fig.shift_origin(xshift="9.6c")
        pygmt.makecpt(cmap="bilbao", series=[-6, 4], log=True)
        fig.grdimage(psd_2D, cmap=True, projection="X8c",
                     region=[xfreqs.min(), xfreqs.max(), yfreqs.min(), yfreqs.max()],
                     frame=["WeSt", "xa0.2f0.05+lk@-x@- [m@+-1@+]", "ya0.2f0.05+lk@-y@- [m@+-1@+]"])
        fig.text(text='b', position="TL", justify="TL",
                 offset='0.1c/-0.1c', font="12p", fill="white")
        fig.colorbar(position="JTC+w8c+o0c/0.3c+h", frame=["a1p+lPower [m@+2@+]"], log=True)
        # 1D power representations, normalised by max power
        # to ensure same scaling for different max amplitude
        fig.shift_origin(xshift="9.6c")
        fig.basemap(projection="X8cl", region=[8e-3, 8e-1, 1e-8, 1e-2],
                frame=["wESt", "xa1f3p+lWavenumber [km@+-1@+]", "ya1f1p+lNormalised Power [m@+2@+/m@+2@+]"])
        fig.plot(x=psd_collapsed[:, 0], y=psd_collapsed[:, 1]/np.max(radial_power),
                 style="c0.05c", fill='grey', transparency="50", no_clip=True, label="Collapsed @~\106@~(k)")
        # fig.plot(x=np.arange(0.7, dim, 0.1),
        #          y=(y_max/100)*np.arange(0.7, dim, 0.1)**exponent, pen="1p,black,--")
        fig.plot(x=bin_data[1:], y=bin_mean_data/np.max(radial_power), style="c0.2c",
                 fill=None, pen="1p,black", no_clip=True, label="Mean of binned power")
        # fig.plot(x=((radial_freq*100)*2+1), y=radial_power, pen="1p,darkred")
        fig.plot(x=radial_freq, y=radial_power/np.max(radial_power), pen="1p,black", no_clip=True,
                 label="Radially averaged power")
        fig.legend()
        fig.plot(x=np.arange(0.01, 0.02, 0.001),
                 y=(5e-8/0.01**0)*np.arange(0.01, 0.02, 0.001)**0, pen="0.5p,gray20")
        # fig.plot(x=np.arange(0.01, 0.02, 0.001),
        #          y=(5e-8/0.01**-1)*np.arange(0.01, 0.02, 0.001)**-1, pen="0.5p,pink")
        fig.plot(x=np.arange(0.01, 0.02, 0.001),
                 y=(5e-8/0.01**-2)*np.arange(0.01, 0.02, 0.001)**-2, pen="0.5p,red")
        fig.plot(x=np.arange(0.01, 0.02, 0.001),
                 y=(5e-8/0.01**1)*np.arange(0.01, 0.02, 0.001)**1, pen="0.5p,blue")
        fig.text(text="k@+1@+", x=0.021, y=1e-7, font="8p", justify="ML")
        fig.text(text="k@+0@+", x=0.021, y=5e-8, font="8p", justify="ML")
        #fig.text(text="k@+-1@+", x=0.021, y=2.8e-8, font="8p", justify="ML")
        fig.text(text="k@+-2@+", x=0.021, y=1.6e-8, font="8p", justify="ML")
        # Add top axes labels
        fig.basemap(projection="X-8cl/8cl", region=[1.25, 125, 1e-2, 1e8],
                    frame=["N", "xa1f3p+lWavelength [km]"])
        fig.text(text='c', position='TL', justify='TL',
                 offset='0.1c/-0.1c', font="12p", fill='white')
    # fig.show()

    if save:
        if savedir is None:
            print("No save directory provided, saving in current directory as tmp.png")
            fig.savefig('tmp.png', dpi=400)
        else:
            fig.savefig(f'{savedir}/random_{colour}_{peak_amp}m_{seed_}.png', dpi=400)

    return None


# Plot noise
print("### Plotting noise figure and saving ###")
setup_r = {'colour': 'red',
         'max_amp': max_amp,
         'seed': seed}
red_xr = xr.DataArray(red)
red_psd_xr = xr.DataArray(red_psd, coords=[x_freqs, y_freqs],
                          dims=['x frequency [m$^{-1}$]', 'y frequency [m$^{-1}$]'])
plot_noise_and_power(red_xr, red_psd_xr, red_psd_collapsed,
                     red_radial, freq_r, bins_, red_bin_means,
                     setup_r, save=True, savedir=red_savedir+'/figs')
setup_w = {'colour': 'white',
         'max_amp': max_amp,
         'seed': seed}
white_xr = xr.DataArray(white)
white_psd_xr = xr.DataArray(white_psd, coords=[x_freqs, y_freqs],
                            dims=['x frequency [m$^{-1}$]', 'y frequency [m$^{-1}$]'])
plot_noise_and_power(white_xr, white_psd_xr, white_psd_collapsed,
                     white_radial, freq_w, bins_, white_bin_means,
                     setup_w, save=True, savedir=white_savedir+'/figs')
setup_b = {'colour': 'blue',
         'max_amp': max_amp,
         'seed': seed}
blue_xr = xr.DataArray(blue)
blue_psd_xr = xr.DataArray(blue_psd, coords=[x_freqs, y_freqs],
                           dims=['x frequency [m$^{-1}$]', 'y frequency [m$^{-1}$]'])
plot_noise_and_power(blue_xr, blue_psd_xr, blue_psd_collapsed,
                     blue_radial, freq_b, bins_, blue_bin_means,
                     setup_b, save=True, savedir=blue_savedir+'/figs')

print("### DONE ### ")
