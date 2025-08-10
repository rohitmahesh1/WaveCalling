import subprocess
from pathlib import Path
from typing import Union, List
import ast
import numpy as np

# Directory containing the Mathematica/WolframScript kymobutler scripts
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "kymobutler_scripts"


def _parse_mathematica_arrays(stdout_text: str) -> List[np.ndarray]:
    """
    Extract the big Mathematica list of coordinate lists from stdout.

    The Wolfram script prints something like {{{{x,y},...}, {{...}}, ...}} plus
    other messages. We:
      1) Find the first '{{{' and the last '}}}' after it
      2) Convert braces -> brackets
      3) ast.literal_eval to a Python list
      4) Normalize each item to a 2-col ndarray of shape (N, 2)
    """
    start = stdout_text.find("{{{")
    if start == -1:
        return []

    # try to find the *last* closing '}}}' (or '}}}}') after start
    end3 = stdout_text.rfind("}}}")
    end4 = stdout_text.rfind("}}}}")
    end = max(end3, end4)
    if end == -1 or end <= start:
        return []

    block = stdout_text[start:end+3]  # include the '}}}'
    # Wolfram lists -> Python lists
    py_list_text = block.replace("{", "[").replace("}", "]")

    try:
        data = ast.literal_eval(py_list_text)
    except Exception:
        # Some scripts wrap coordinates like [[[ [x,y], ... ]]]; try a looser approach
        return []

    arrays = []
    for item in data:
        # Some outputs nest an extra level: [ [ [x,y], ... ], meta? ... ]
        # Heuristic: if item[0] looks like a list of 2 numbers, use that.
        try:
            cand = np.array(item, dtype=float)
        except Exception:
            continue

        if cand.ndim == 2 and cand.shape[1] == 2:
            arrays.append(cand)
            continue

        # Try item[0]
        try:
            cand0 = np.array(item[0], dtype=float)
            if cand0.ndim == 2 and cand0.shape[1] == 2:
                arrays.append(cand0)
                continue
        except Exception:
            pass

        # As a last resort, flatten to Nx2 if possible
        flat = cand.reshape(-1, 2) if cand.size % 2 == 0 else None
        if flat is not None and flat.ndim == 2 and flat.shape[1] == 2:
            arrays.append(flat)

    return arrays


def run_kymobutler(
    heatmap_path: Union[str, Path],
    min_length: int = 30,
    verbose: bool = False,
    script_name: str = "Run_Kymobutler.wls",  # matches your earlier setup
) -> Path:
    """
    Invoke the KymoButler WolframScript on a heatmap image, parse stdout for
    track coordinates, and save each track as .npy under:
        <heatmap_dir>/<heatmap_stem>/kymobutler_output/

    Returns the base_dir (parent folder that contains kymobutler_output/).
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    out_dir = base_dir / "kymobutler_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = SCRIPT_DIR / script_name
    cmd = [
        "wolframscript", "-script",
        str(script),
        str(heatmap_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Always keep stdout/stderr for parsing/logging; show only if verbose
    if verbose:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

    # This raises if wolframscript returned non-zero
    proc.check_returncode()

    # Parse Mathematica stdout into arrays and write .npy files
    arrays = _parse_mathematica_arrays(proc.stdout or "")
    saved = 0
    for i, arr in enumerate(arrays):
        # Filter out very short traces right here, if desired
        if arr.shape[0] < min_length:
            continue
        np.save(out_dir / f"{i}.npy", arr)
        saved += 1

    if verbose:
        print(f"[kymo_interface] Saved {saved} track(s) to {out_dir}")

    return base_dir


def preprocess_image(
    image_path: Union[str, Path],
    verbose: bool = False
) -> Path:
    """
    Run the Mathematica preprocessing script to prepare a raw kymograph image.

    Returns the processed image path in the 'proc' subfolder.
    """
    image_path = Path(image_path)
    proc_dir = image_path.parent / image_path.stem / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)
    script = SCRIPT_DIR / "Run_Kymobutler_Save_Image.wls"
    cmd = [
        "wolframscript", "-script",
        str(script),
        str(proc_dir),
        str(image_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    proc.check_returncode()

    processed = proc_dir / f"{image_path.stem}_processed_2.png"
    if not processed.exists():
        raise FileNotFoundError(f"Processed image not found: {processed}")
    return processed
