import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
from tqdm import tqdm

def merge_close_indices(indices, data, min_distance):
    """
    Merges indices that are too close to each other based on a specified minimum distance.

    Args:
        indices (np.array): Array of indices to be merged.
        data (np.array): Data array used for comparison.
        min_distance (int): Minimum allowed distance between indices.

    Returns:
        np.array: Array of merged indices.
    """
    if len(indices) < 2:
        return indices
    
    merged_indices = []
    current_group = [indices[0]]

    for i in indices[1:]:
        if i - current_group[-1] < min_distance:
            current_group.append(i)
        else:
            # Select the index with the most extreme value
            best_index = max(current_group, key=lambda x: data[x])
            merged_indices.append(best_index)
            current_group = [i]
    
    # Handle the last group
    best_index = max(current_group, key=lambda x: data[x])
    merged_indices.append(best_index)

    return np.array(merged_indices)

def detect_extrema(data, window_length=10, polyorder=3, min_distance=10):
    """
    Detects local maxima and minima in a given noisy data array, merging close extrema.

    Args:
        data (np.array): The input data array.
        window_length (int): The length of the window for the smoothing filter.
        polyorder (int): The order of the polynomial used in the smoothing filter.
        min_distance (int): Minimum distance between extrema to avoid close duplicates.

    Returns:
        merged_extrema_indices (np.array): Indices of the merged local maxima and minima.
    """
    # Smooth the data
    if window_length % 2 == 0:  # Savitzky-Golay filter requires an odd window length
        window_length += 1

    smoothed_data = savgol_filter(data, window_length, polyorder)

    # Find local maxima and minima
    maxima_indices = argrelextrema(smoothed_data, np.greater)[0]
    minima_indices = argrelextrema(smoothed_data, np.less)[0]

    # Combine maxima and minima indices
    combined_indices = np.sort(np.concatenate((maxima_indices, minima_indices)))

    # Merge close extrema
    merged_extrema_indices = merge_close_indices(combined_indices, smoothed_data, min_distance)
    
    return merged_extrema_indices, smoothed_data

import numpy as np
import scipy.optimize
from scipy.signal import lombscargle

def fit_sin(tt, yy, x_ext, y_ext):
    '''Fit sin to the input time sequence with constraints that (x_ext, y_ext) is a local maximum and the function passes through (x_ext, y_ext), and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc", adjusted for non-uniform sampling'''
    tt = np.array(tt)
    yy = np.array(yy)

    # Estimate frequency using Lomb-Scargle
    freqs = np.linspace(0.01, 1, 10000)  # Frequency range and density may need to be adjusted
    power = lombscargle(tt, yy, freqs)
    guess_freq = freqs[np.argmax(power)]

    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess_phase = 0.
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, guess_phase, guess_offset])

    # Sinusoidal function
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    # Objective function to minimize (mean squared error)
    def objective(params):
        return np.sum((sinfunc(tt, *params) - yy) ** 2)

    # Constraint for zero derivative at x_ext (ensure local maximum or minimum)
    def derivative_constraint(params):
        A, w, p, c = params
        return A * w * np.cos(w * x_ext + p)

    # Constraint for the function value at x_ext being y_ext
    def value_constraint(params):
        A, w, p, c = params
        return sinfunc(x_ext, A, w, p, c) - y_ext

    # Use scipy.optimize.minimize with constraints
    cons = [
        {'type': 'eq', 'fun': derivative_constraint},
        {'type': 'eq', 'fun': value_constraint}
    ]
    result = scipy.optimize.minimize(objective, guess, constraints=cons, options={"maxiter":10000})

    if result.success:
        A, w, p, c = result.x
        f = w / (2. * np.pi)
        fitfunc = lambda t: A * np.sin(w * t + p) + c
        return {
            "amp": A, "omega": w, "phase": p, "offset": c,
            "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": None, "rawres": (guess, result.x, None)
        }
    else:
        raise RuntimeError("Optimization failed: " + result.message)

# Usage example
# result = fit_sin(time_data, amplitude_data, extremum_time_point, extremum_amplitude)

def perform_local_sin_regressions(x, y_smoothed, extrema_indices, radius, window_length=7, polyorder=3):
    x = np.array(x)
    y_smoothed = np.array(y_smoothed)
    """
    Performs local sinusoidal regressions around each extremum index within a given radius,
    including the extremum's coordinates, with preliminary Savitzky-Golay smoothing.

    Args:
        x (np.array): x-axis data points.
        y_smoothed (np.array): y-axis data points.
        extrema_indices (np.array): Indices of extrema.
        radius (float): Radius within which to perform regression around each extremum.
        window_length (int): The length of the window for the Savitzky-Golay filter.
        polyorder (int): The order of the polynomial used in the Savitzky-Golay filter.

    Returns:
        list: A list of dictionaries, each containing the extremum index, extremum coordinates,
              the slice of x and y used, and the fitted model.
    """
    # Apply Savitzky-Golay filter to the entire dataset
    #y_smoothed = savgol_filter(y, window_length, polyorder) if len(y) > window_length else y

    results = []
    for index in extrema_indices:
        min_x = x[index] - radius
        max_x = x[index] + radius
        valid_indices = (x >= min_x) & (x <= max_x)
        x_slice = x[valid_indices]
        y_slice = y_smoothed[valid_indices]

        if len(x_slice) > 0 and len(y_slice) > 0:
            fitted_model = fit_sin(x_slice, y_slice, x[index], y_smoothed[index])
            results.append({
                "extremum_index": index,
                "extremum_coordinates": (x[index], y_smoothed[index]),
                "x_slice": x_slice,
                "y_slice": y_slice,
                "fitted_model": fitted_model
            })
            
    return results

def plot_individual_fitted_models(x, y, regression_results, y_smoothed):
    """
    Plots a quarter period on either side of each extremum for the fitted sinusoidal models.

    Args:
        x (np.array): Original x-axis data points.
        y (np.array): Original y-axis data points.
        regression_results (list): List of regression result dictionaries.
        y_smoothed (np.array): Smoothed y-axis data points.
    """
    #print("x", x)
    #print("x_len", len(x))
    for result in regression_results:

        #plt.figure(figsize=(8, 4))
        model = result['fitted_model']
        x_slice = result['x_slice']
        #print("x_slice", x_slice)
        x_slice = np.linspace(x_slice[0], x_slice[-1], int((x_slice[-1]-x_slice[0])/0.1))
        extremum_x, extremum_y = result['extremum_coordinates']
        x_slice_extremum_index = len(np.where(x_slice < extremum_x)[0])
        #print("x_slice_extremum_index", x_slice_extremum_index)

        #print("extremum coords", result['extremum_coordinates'])
        y_fitted = model['amp'] * np.sin(model['omega'] * x_slice + model['phase']) + model['offset']
        
        # Calculate quarter-period plot range
        period = abs(2 * np.pi / model['omega'])
        x_slice_length = x_slice[-1] - x_slice[0]
        n_indices_in_period = period / x_slice_length * (len(x_slice) - 1)
        quarter_period_idx = n_indices_in_period // 4
        #print("quarter_period", quarter_period_idx)
        #quarter_period_idx = int(period / 4 / np.diff(x_slice).mean())  # Quarter-period index count
        #extremum_idx = np.argmin(np.abs(x_slice - extremum_x))
        #print("extremum_idx", extremum_idx)
        # Define slice around the extremum for quarter period on both sides
        #print("result ex idx", result['extremum_index'])
        plot_slice = slice(int(max(0, x_slice_extremum_index - quarter_period_idx)), min(x_slice_extremum_index + int(quarter_period_idx), (len(x_slice)-1)))
        #print("plot_slice", plot_slice)
        """
        plt.plot(x, y, label='Original Data', color='grey', alpha=0.5)
        plt.plot(x, y_smoothed, label='Smoothed Data', color='blue', alpha=0.7)
        plt.plot(x_slice[plot_slice], y_fitted[plot_slice], 'r-', label='Fitted Sinusoid (Quarter Periods)')
        plt.scatter(extremum_x, extremum_y, color='magenta', s=100, label='Extremum', zorder=5)"""

        #plt.plot(y, x, label='Original Data', color='grey', alpha=0.5)
        #plt.plot(y_smoothed, x, label='Smoothed Data', color='blue', alpha=0.7)
        plt.plot(y_fitted[plot_slice], x_slice[plot_slice], 'o-', linewidth=1, markersize=1)
        #plt.scatter(extremum_y, extremum_x, color='magenta', s=100, label='Extremum', zorder=5)



Y_LIM_RANGE = 60
def detect_and_plot_extrema(x, y, filename):
    x = np.array(x)
    y = np.array(y)
    merged_extrema_indices, smoothed_y = detect_extrema(y, min_distance=15)  # Set min_distance as needed
    if len(merged_extrema_indices) == 0:
        print("No extrema detected")
        return None, None, None
    
    regression_results = perform_local_sin_regressions(x, smoothed_y, merged_extrema_indices, 30)

    y_min, y_max = y.min(), y.max()
    y_smoothed_min, y_smoothed_max = smoothed_y.min(), smoothed_y.max()
    y_extrema = [y_min, y_max, y_smoothed_min, y_smoothed_max]
    y_range = max(y_extrema) - min(y_extrema)
    padding = (Y_LIM_RANGE - y_range) / 2
    plot_y_min, plot_y_max = min(y_extrema) - padding, max(y_extrema) + padding
    # Plot results
    plt.clf()
    plt.figure(figsize=(10, 6))
    #plt.plot(x, y, label='Noisy Data')
    #plt.plot(x, smoothed_y, color='red', label='Smoothed Data')
    plt.plot(y, x, label='Noisy Data')
    plt.plot(smoothed_y, x, color='red', label='Smoothed Data')
    #print("merged extrema indices", merged_extrema_indices)
    #plt.scatter(x[merged_extrema_indices], smoothed_y[merged_extrema_indices], color='green', s=100, label='Merged Extrema')
    plt.scatter(smoothed_y[merged_extrema_indices], x[merged_extrema_indices], color='green', s=100, label='Merged Extrema')
    plot_individual_fitted_models(x, y, regression_results, smoothed_y)
    plt.xlim(plot_y_min, plot_y_max)

    plt.title('Local Extrema Detection in Noisy Data')
    #plt.title('Filtered Data')

    plt.legend()
    #plt.show()
    plt.savefig(filename)
    
    #print("disp", smoothed_y.max() - smoothed_y.min())
    return merged_extrema_indices, smoothed_y, y

#extrema_indices, y_smoothed = detect_and_plot_extrema(x_real, y_real)
#detect_and_plot_extrema(x, y)

