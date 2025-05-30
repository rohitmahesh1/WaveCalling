import fire
from pathlib import Path
from csv_to_heatmap import filter_csv_and_generate_heatmap_2
from call_waves import run_kymobutler, process_image

def run(input_path: str, min_trace_length: int = 30, push_to_drive: bool = False):
    """
    input_path: CSV or image file
    min_trace_length: minimum length of traces to analyze
    push_to_drive: if True, tries to find client_secret*.json in cwd and upload;
                   otherwise saves results locally.
    """
    p = Path(input_path)
    if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        # directly process image
        process_image(str(p), min_length=min_trace_length, push_to_drive=push_to_drive)
    else:
        # first generate heatmap from CSV, get dims
        out, rows, cols = filter_csv_and_generate_heatmap_2(str(p))
        print("Rows:", rows, "Cols:", cols)
        # then run KymoButler on heatmap
        run_kymobutler(str(out), rows, cols, min_length=min_trace_length, push_to_drive=push_to_drive)

if __name__ == "__main__":
    fire.Fire(run)
