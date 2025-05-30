import subprocess
import ast
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sine_regression
from google_utils import create_google_sheet_with_dataframe
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent / "kymobutler_scripts"

def _find_client_secret():
    """
    Look for a file named client_secret*.json in the current working directory.
    """
    for p in Path('.').glob('client_secret*.json'):
        return str(p.resolve())
    return None

def parse_output(output):
    # Extract the {{{…}}}} block and turn into Python lists
    start = output.find("{{{")
    end = output.find("}}}}") + 4
    block = output[start:end].replace('{', '[').replace('}', ']')
    try:
        results = ast.literal_eval(block[:-1])
        arrays = []
        for item in results:
            arr = item[0] if len(np.array(item[0]).shape) > 1 else item
            arrays.append(arr)
        return arrays
    except SyntaxError:
        print("Error parsing output")
        return []

def run_kymobutler(file_path, num_rows, num_cols, min_length=30, push_to_drive=False):
    """
    1) Runs the WolframScript on file_path
    2) Filters out traces shorter than min_length
    3) Optionally pushes results to Google Sheets (or saves locally)
    """
    # build the script path
    script = SCRIPT_DIR / "Run_Kymobutler_updated_greyscale.wls"
    cmd = f'wolframscript -script "{script}" "{file_path}"'
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if err:
        print("WolframScript errors:\n", err.decode())

    # parse and process the arrays
    arrays = parse_output(out.decode())
    base = Path(file_path).parent / Path(file_path).stem
    (base / "sine_regression").mkdir(parents=True, exist_ok=True)
    (base / "kymobutler_output").mkdir(parents=True, exist_ok=True)

    lengths, widths, dfs = [], [], []
    for i, arr in tqdm(enumerate(arrays), total=len(arrays)):
        arr = np.array(arr)
        lengths.append(arr.shape[0])
        if arr.shape[0] < min_length:
            continue
        np.save(base / "kymobutler_output" / f"{i}.npy", arr)

        arr_t = arr.T
        _, _, y, df = sine_regression.detect_and_plot_extrema(
            arr_t[0], arr_t[1],
            str(base / "sine_regression" / f"{i}.png"),
            i
        )
        if y is not None:
            widths.append(y.max() - y.min())
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if push_to_drive:
        cs = _find_client_secret()
        if not cs:
            raise FileNotFoundError("No client_secret*.json in cwd")
        name = f'Wave Data {Path(file_path).stem}'
        create_google_sheet_with_dataframe(cs, name, combined, 'Wave Data')
    else:
        csv_path = base / f"{Path(file_path).stem}_wave_data.csv"
        combined.to_csv(csv_path, index=False)
        print(f"Data saved locally to {csv_path}")

    # length histogram
    lengths = np.array(lengths)
    plt.figure()
    ax = sns.histplot(lengths)
    plt.axvline(min_length, color='r', label='cutoff')
    plt.axvline(lengths[lengths > min_length].mean(), color='b', label='mean')
    plt.legend()
    ax.set(title="Wave lengths", xlabel="length", ylabel="count")
    plt.savefig(base / (Path(file_path).stem + "_length_distribution.png"))
    plt.close()

    # width histogram
    widths = np.array(widths)
    plt.figure()
    ax = sns.histplot(widths)
    plt.axvline(widths.mean(), color='b', label='mean width')
    plt.legend()
    ax.set(title="Wave widths", xlabel="width", ylabel="count")
    plt.savefig(base / (Path(file_path).stem + "_width_distribution.png"))
    plt.close()


def process_image(image_path, min_length=30, push_to_drive=False):
    """
    1) Runs WolframScript to preprocess image
    2) Detects colored lines and computes extrema per line
    3) Optionally pushes results to Google Sheets (or saves locally)
    """
    path = Path(image_path)
    proc_dir = path.parent / path.stem / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # invoke the Save_Image script from the new folder
    script = SCRIPT_DIR / "Run_Kymobutler_Save_Image.wls"
    cmd = f'wolframscript -script "{script}" "{proc_dir}" "{image_path}"'
    subprocess.Popen(cmd, shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE).communicate()

    img_file = proc_dir / (path.stem + "_processed_2.png")
    img = Image.open(img_file)
    arr = np.array(img)
    coords = np.column_stack(np.nonzero(np.any(arr != [0,0,0], axis=-1)))
    df_coords = pd.DataFrame(coords, columns=["Y","X"])
    df_coords['Color'] = list(map(tuple, arr[coords[:,0], coords[:,1]]))

    # sort lines by their top-most pixel
    lines = sorted(
        [(c, df_coords[df_coords['Color']==c]['Y'].min())
         for c in df_coords['Color'].unique()],
        key=lambda x: x[1]
    )

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    sine_dir = proc_dir / "sine_regression"
    sine_dir.mkdir(exist_ok=True)

    dfs = []
    for idx, (color, y0) in enumerate(lines, start=1):
        subset = df_coords[df_coords['Color']==color]
        fname = sine_dir / f"{path.stem}_{idx}_sine_regression.png"
        _, _, y, df = sine_regression.detect_and_plot_extrema(
            subset['Y'], subset['X'], str(fname), idx
        )
        draw.text((subset['X'].iloc[0], y0), str(idx),
                  fill="white", font=font)
        if y is not None:
            dfs.append(df.fillna(''))

    ann_path = sine_dir / f"{path.stem}_annotated.png"
    annotated.save(ann_path)
    print(f"Annotated image saved to {ann_path}")

    combined = pd.concat(dfs, ignore_index=True)
    if push_to_drive:
        cs = _find_client_secret()
        if not cs:
            raise FileNotFoundError("No client_secret*.json in cwd")
        name = f'Wave Data {path.stem}'
        create_google_sheet_with_dataframe(cs, name, combined, 'Wave Data')
    else:
        csv_path = proc_dir / f"{path.stem}_line_data.csv"
        combined.to_csv(csv_path, index=False)
        print(f"Line data saved locally to {csv_path}")
