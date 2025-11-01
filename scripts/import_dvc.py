import argparse as ap
from pathlib import Path

TEST_DATA_PATH = Path(__file__).parents[1] / "tests" / "test_nn" / "data"

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Import DVC files.")
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to the data directory containing DVC files.",
    )
    args = parser.parse_args()
    data_path = args.data_path

    # List of DVC files to import
    dvc_files = data_path.glob("*.npz.dvc")

    for dvc_file in dvc_files:
        dvc_path = Path(dvc_file)
        target_path = TEST_DATA_PATH / dvc_path.name
        print(f"Importing {dvc_path} to {target_path}")
        target_path.write_text(dvc_path.read_text())
