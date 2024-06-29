import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema

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

# Example usage
np.random.seed(300)
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.5, 100)

def detect_and_plot_extrema(x, y, filename):
    x = np.array(x)
    y = np.array(y)
    merged_extrema_indices, smoothed_y = detect_extrema(y, min_distance=15)  # Set min_distance as needed
    if len(merged_extrema_indices) == 0:
        print("No extrema detected")
        return 
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Noisy Data')
    plt.plot(x, smoothed_y, color='red', label='Smoothed Data')
    print("merged extrema indices", merged_extrema_indices)
    plt.scatter(x[merged_extrema_indices], smoothed_y[merged_extrema_indices], color='green', s=100, label='Merged Extrema')
    plt.title('Local Extrema Detection in Noisy Data')
    #plt.title('Filtered Data')

    plt.legend()
    #plt.show()
    plt.savefig(filename)
    return merged_extrema_indices, smoothed_y

#extrema_indices, y_smoothed = detect_and_plot_extrema(x_real, y_real)
#detect_and_plot_extrema(x, y)

