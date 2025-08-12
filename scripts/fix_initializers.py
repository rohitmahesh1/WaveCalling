# scripts/fix_initializers.py
import onnx
from onnx import helper

def remove_initializers_from_inputs(src, dst):
    model = onnx.load(src)

    init_names = {i.name for i in model.graph.initializer}
    # collect the inputs we want to keep
    keep_inputs = [vi for vi in model.graph.input if vi.name not in init_names]

    # protobuf repeated fields don't support slice assignment
    model.graph.ClearField("input")
    model.graph.input.extend(keep_inputs)

    onnx.checker.check_model(model)
    onnx.save(model, dst)

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Remove initializers from graph inputs for all .onnx files in a directory."
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default="exports",
        help="Directory containing .onnx files (default: exports)",
    )
    parser.add_argument(
        "--ext",
        default="onnx",
        help="File extension to match (default: onnx)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without writing changes.",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        # in case your folder is named 'export' as in the example comment
        alt = Path("export")
        if alt.exists():
            root = alt
        else:
            sys.exit(f"Directory not found: {root.resolve()}")

    files = sorted(p for p in root.glob(f"*.{args.ext}") if p.is_file())
    if not files:
        print(f"No .{args.ext} files found in {root}")
        sys.exit(0)

    print(f"Processing {len(files)} file(s) in {root}...\n")
    failures = 0
    for p in files:
        try:
            if args.dry_run:
                print(f"[DRY RUN] Would fix: {p.name}")
            else:
                remove_initializers_from_inputs(str(p), str(p))  # in-place
                print(f"✔ Fixed: {p.name}")
        except Exception as e:
            failures += 1
            print(f"✖ Failed: {p.name} -> {e}", file=sys.stderr)

    if failures:
        sys.exit(f"{failures} file(s) failed.")
