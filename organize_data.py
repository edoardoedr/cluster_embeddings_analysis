"""
Riorganizza i CSV in Data/Features_new/ (formato piatto
'<DATASET>_<MODEL>_<LABEL>.csv') nella struttura a cartelle attesa dagli
script di analisi:

    Data/shap_filtered_datasets/<LABEL>/<MODEL>/<dataset>_<LABEL>_dataset.csv

I file originali non vengono toccati (copia, non spostamento).
"""
import re
import shutil
from pathlib import Path

SOURCE_DIR = Path(__file__).parent / "Data" / "Features_new"
DEST_ROOT = Path(__file__).parent / "Data" / "shap_filtered_datasets"

DATASET_RENAME = {
    "CHAPMAN": "chapman_ningbo",
    "COD15": "cod15",
    "GEORGIA": "georgia",
    "ptbxl": "ptbxl",
}

FILENAME_RE = re.compile(r"^(?P<dataset>[^_]+)_(?P<model>.+)_(?P<label>[A-Za-z]+)$")


def parse_filename(stem):
    match = FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"Nome file inatteso: {stem}")
    return match.group("dataset"), match.group("model"), match.group("label")


def main():
    csv_files = sorted(SOURCE_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"Nessun CSV trovato in {SOURCE_DIR}")

    copied = 0
    for csv_path in csv_files:
        raw_dataset, model, label = parse_filename(csv_path.stem)

        if raw_dataset not in DATASET_RENAME:
            raise ValueError(f"Dataset non mappato: {raw_dataset} (file: {csv_path.name})")
        dataset = DATASET_RENAME[raw_dataset]

        dest_dir = DEST_ROOT / label / model
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{dataset}_{label}_dataset.csv"

        shutil.copy2(csv_path, dest_path)
        print(f"  {csv_path.name}  ->  {dest_path.relative_to(DEST_ROOT.parent)}")
        copied += 1

    print(f"\nCopiati {copied}/{len(csv_files)} file in {DEST_ROOT}")


if __name__ == "__main__":
    main()
