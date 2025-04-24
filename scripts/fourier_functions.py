import warnings
import numpy as np


def calculate_2d_ft(data_):
    """
    Calculate a 2D Fourier transform of a 2d nd array
    
    Parameters
    ----------
    data: ndarray
        The array upon which to calculate the Fourier Transform

    Returns
    -------
    ft: ndarray
        The 2D Fourier transform
    """
    ft = np.fft.ifftshift(data_)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return ft


def calculate_1d_ft(data):
    """
    Calculate a 1D Fourier transform on real input

    Parameters
    -----------
    data: one dimensional ndarray
        1D array upon which to calculate the Fourier transform
    power_spectrum: Boolean

    Returns
    --------
    ft: ndarray
        Real and positive components for the 1D Fourier transform of data
    """
    # Check array is one dimensional
    assert data.ndim == 1, "Array should be one-dimensional"

    # Calculate Fourier transforms
    ft = np.fft.rfft(data)

    return ft


def calculate_1d_power_spectrum(ft_data, distance):
    """
    Calculate a 1D power spectrum on a Fourier transform

    Parameters
    -----------
    ft_data: nd.array
        One dimensional array of a Fourier transform
    distance: integer
        Length of original data (prior to Fourier transform)
    
    Returns
    ----------
    psd: ndarray
        Real and positive components of the 1D power spectrum
    freqs: ndarray
        Frequencies corresponding to the samples for the Fourier transform
    """
    # Normalise. This subset contains only the positive frequencies as per np.fft documentation
    ft_data = (2 / distance) * np.abs(ft_data[1:int(distance/2)+1])

    # Calculate power
    psd = np.abs(ft_data)**2
    freqs =  np.fft.rfftfreq(distance)
    freqs = freqs[1:int(distance/2)+1]
    return psd, freqs


def calculate_average_power_1d(data, axis=1):
    """
    Calculate the average power across an axis of a 2D array.

    Paramaters
    ----------
    data_: 2D nd.array of real values
    axis: integer of 0 (y) or 1 (x) corresponding to desired axis

    Returns
    ---------
    pwr: 1D array of power spectrum
    """
    # Check axis is 0 or 1
    assert axis in [0, 1], "axis must equal 0 or 1"

    # Calculate Fourier transform along the chosen axis
    ft_ = np.fft.rfft(data, axis=axis)

    # Get length of axis
    dist = data.shape[axis]

    # Normalise Fourier transform
    if axis == 0:
        ft_ = (2 / dist) * abs(ft_[1:int(dist/2)+1, :])
    else:
        ft_ = (2 / dist) * abs(ft_[:, 1:int(dist/2)+1])

    # Sum power along the second axis
    pwr = ft_.real*ft_.real + ft_.imag*ft_.imag
    if axis == 0:
        pwr = pwr.sum(axis=1)/pwr.shape[1]
    else:
        pwr = pwr.sum(axis=0)/pwr.shape[0]

    return pwr


# from https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/spectral.py#L100
def rapsd(field, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs):
    """
    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.

    Parameters
    ----------
    field: array_like
        A 2d array of shape (m, n) containing the input field.
    fft_method: object
        A module or object implementing the same methods as numpy.fft and
        scipy.fftpack. If set to None, field is assumed to represent the
        shifted discrete Fourier transform of the input field, where the
        origin is at the center of the array
        (see numpy.fft.fftshift or scipy.fftpack.fftshift).
    return_freq: bool
        Whether to also return the Fourier frequencies.
    d: scalar
        Sample spacing (inverse of the sampling rate). Defaults to 1.
        Applicable if return_freq is 'True'.
    normalize: bool
        If True, normalize the power spectrum so that it sums to one.

    Returns
    -------
    out: ndarray
        One-dimensional array containing the RAPSD. The length of the array is
        int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: ndarray
        One-dimensional array containing the Fourier frequencies.

    References
    ----------
    :cite:`RC2011`
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    # Compute centred array
    if m % 2 == 1:
        s1 = np.s_[-int(m / 2) : int(m / 2) + 1]
    else:
        s1 = np.s_[-int(m / 2) : int(m / 2)]

    if n % 2 == 1:
        s2 = np.s_[-int(n / 2) : int(n / 2) + 1]
    else:
        s2 = np.s_[-int(n / 2) : int(n / 2)]

    yc, xc = np.ogrid[s1, s2]

    # Calculate radial grid
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size**2
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


# Plot 2D Power spectra collapsed into 1D scatter, as per Perron et al 2008 Fig 4a,b
# Radial frequency vs mean-squared amplitude
def collapse_2d_psd_to_1d(power_spectrum, return_freq=True):
    """
    Convert a two-dimensional power spectrum into a 
    one dimensional dataset of power and corresponding
    radial frequencies.

    Parameters
    --------------
    power_spectrum: nd.array
                    2d nd.array of a power spectrum. Must have already been fftshifted
                    (ie central coordinate must be zero frequency).
    return_freq: bool, optional
                 If True, returns frequencies (wavenumbers) [m^-1] as the first column.
                 If False, returns radial distances [m] as the first column

    Returns
    -------------
    psd_data: Two-dimensional array where each item contains [radial frequency or distance, power]
    """

    # Ensure power spectrum is 2D
    assert power_spectrum.ndim == 2, "Power spectrum must have 2 dimensions."

    # Find the central coordinate
    y, x = np.indices(power_spectrum.shape) # array of indices
    if x.max() % 2 == 1:  # Deal with odd/even dimensions
        xc = (x.max() - x.min())/2. + 1
    else:
        xc = (x.max() - x.min())/2.
    if y.max() % 2 == 1:
        yc = (y.max() - y.min())/2. + 1
    else:
        yc = (y.max() - y.min())/2.
    center = np.array([xc, yc])

    # Calculate grid of wavenumbers
    if return_freq:
        ny, nx = power_spectrum.shape
        fx = np.fft.fftshift(np.fft.fftfreq(nx, 1))  # assumes a grid spacing of 1 km
        fy = np.fft.fftshift(np.fft.fftfreq(ny, 1))

    # Loop over PSD array, calculate radial distances from centre and store with power
    psd_data  = []
    for ii in range(power_spectrum.shape[0]):
        for jj in range(power_spectrum.shape[1]):
            if return_freq:
                radial_freq = np.sqrt((fy[ii]**2 + fx[jj]**2))
                psd_data.append([radial_freq, power_spectrum[ii, jj]])
            else:
                radial_dist = np.sqrt((jj - center[1])**2 + (ii - center[0])**2)
                psd_data.append([radial_dist, power_spectrum[ii, jj]])
   
    # Convert to array
    psd_data = np.array(psd_data)

    return psd_data


def calculate_averages_in_bins(collapsed_power_spectrum, bins):
    """ 
    Calculate the mean power within a bin for a collapsed power spectrum

    Parameters
    -------------------
    collapsed_power_spectrum: Two-dimensional array of frequency, power data.
    E.g. output from collapse_2d_psd_to_1d function
    bins: array of bins, e.g. logarithmically spaced.

    Returns
    ----------
    bin_means: Mean values of power per bin
    """

    # Get indices of the power data corresponding to each bin
    digitized = np.digitize(collapsed_power_spectrum[:, 0], bins)

    # Calculate mean per bin, catch runtime warning for mean of an empty bin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bin_means = [collapsed_power_spectrum[digitized == ii, 1].mean() for ii in range(1, len(bins))]

    return np.array(bin_means)
