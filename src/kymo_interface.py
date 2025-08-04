import subprocess
from pathlib import Path
from typing import Union

# Directory containing the Mathematica/WolframScript kymobutler scripts
SCRIPT_DIR = Path(__file__).parent / "kymobutler_scripts"


def run_kymobutler(
    heatmap_path: Union[str, Path],
    min_length: int = 30,
    verbose: bool = False
) -> Path:
    """
    Invoke the KymoButler WolframScript on a heatmap image.

    Args:
        heatmap_path: Path to the .png/.jpg heatmap generated from CSV.
        min_length: Minimum trace length for downstream filtering (optional,
            can be passed through to further analyzers).
        verbose: If True, prints script stdout/stderr.

    Returns:
        base_dir: Path object pointing to the base output directory
            (heatmap_path.stem), where 'kymobutler_output' and
            'sine_regression' subfolders live.
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    script = SCRIPT_DIR / "Run_Kymobutler.wls"
    cmd = [
        "wolframscript", "-script",
        str(script),
        str(heatmap_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(proc.stdout)
        print(proc.stderr)
    proc.check_returncode()

    return base_dir


def preprocess_image(
    image_path: Union[str, Path],
    verbose: bool = False
) -> Path:
    """
    Run the Mathematica preprocessing script to prepare a raw kymograph image.

    Args:
        image_path: Path to the raw kymograph image (.png/.jpg).
        verbose: If True, prints script stdout/stderr.

    Returns:
        proc_image: Path to the processed image file in the 'proc' subfolder.
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
        print(proc.stdout)
        print(proc.stderr)
    proc.check_returncode()

    processed = proc_dir / f"{image_path.stem}_processed_2.png"
    if not processed.exists():
        raise FileNotFoundError(f"Processed image not found: {processed}")
    return processed
