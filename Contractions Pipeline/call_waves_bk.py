import fire
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import ast
import sine_regression
import datapoint
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

def parse_output(output):
    # Replace curly braces with square brackets
    #print("output", output)
    output = output[output.find("{{{"): output.find("}}}}")+4]
    formatted_output = output.replace('{', '[').replace('}', ']')
    #print("formatted_output", formatted_output)
    try:
        # Safely evaluate the string as a Python literal
        results = ast.literal_eval(formatted_output[:-1])
        #print("result", results)
        # Convert the result into a list of numpy arrays
        #numpy_arrays = [np.array(sublist) for sublist in result]
        arrays = []
        for result in results:
            #print("result before", result)
            if len(np.array(result[0]).shape) > 1:
                #print("result0", result[0])
                arrays.append(result[0])
            else:
                arrays.append(result)
        #print("arrays", arrays)
        return arrays
    except SyntaxError as e:
        print("Error parsing the output:", e)
        return []




def plot_results(arrays):
    plt.figure(figsize=(10, 6))
    for i, arr in enumerate(arrays):
        #print("type arr", type(arr))
        arr = np.array(arr)
        plt.plot(arr[:, 0], arr[:, 1], marker='o', linestyle='-', label=f'Series {i+1}')
    plt.title('Plot of Points from WolframScript Output')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

def run_kymobutler(file_path, min_length=30):
    # Execute the WolframScript command
    command = f'wolframscript -script Run_KymoButler.wls {file_path}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Decode the output to string
    output = stdout.decode()
    
    # Print stderr if needed to see errors from wolframscript
    if stderr:
        print("Errors:\n", stderr.decode())

    # Parse the output
    arrays = parse_output(output)
    #arrays = datapoint.data

    # Plot the results
    #plot_results(arrays)
    lengths = []
    widths = []
    #for array in arrays: print("array", np.array(array).shape)
    #print("np array", np.array(arrays))
    (Path(file_path).parent / (Path(file_path).stem + "_sine_regression")).mkdir(parents=True, exist_ok=True)
    for i, array in tqdm(enumerate(arrays), total=len(arrays)):
        array = np.array(array)
        lengths.append(array.shape[0])
        if array.shape[0] < min_length:
            continue

        array = array.T
        _, y_smoothed, y = sine_regression.detect_and_plot_extrema(array[0], array[1], f"{Path(file_path).parent / (Path(file_path).stem + "_sine_regression")}/{i}.png")
        if y is not None:
            widths.append(y.max() - y.min())

    """
    lengths
    """
    lengths = np.array(lengths)
    plt.clf()
    ax = sns.histplot(lengths)
    plt.axvline(x=min_length, color='r', label='minimum cutoff length')
    plt.axvline(x=lengths[lengths > min_length].mean(), color='b', label='mean length of waves considered for analysis')
    plt.legend()
    ax.set(title="Distribution of lengths of waves", xlabel="length", ylabel="count")
    ax.get_figure().savefig(Path(file_path).parent / (Path(file_path).stem + "_length_distribution.png")) 

    """
    widths
    """
    widths = np.array(widths)
    plt.clf()
    ax = sns.histplot(widths)
    #plt.axvline(x=min_length, color='r', label='minimum cutoff length')
    plt.axvline(x=widths.mean(), color='b', label='mean width of waves considered for analysis')
    plt.legend()
    ax.set(title="Distribution of widths of waves", xlabel="width", ylabel="count")
    ax.get_figure().savefig(Path(file_path).parent / (Path(file_path).stem + "_width_distribution.png")) 

if __name__ == '__main__':
    fire.Fire(run_kymobutler)
