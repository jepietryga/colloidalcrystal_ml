from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from facet_ml.static.path import STATIC_FOLDER, STATIC_MODELS

ZENODO_RECORD_ID = "14019586"
ZENODO_API_BASE = "https://zenodo.org/api/records"
SAM_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
)


MODEL_DESTINATIONS = {
    "edge_classifier.pickle": Path("edge_classifier.pickle"),
    "bg_segmenter.pickle": Path("bg_segmenter.pickle"),
    "maskrcnn_model.pth": Path("torch") / "maskrcnn_model.pth",
    "RF_C_MC.sav": Path("2023_11_models_length-agnostic") / "RF_C_MC.sav",
    "RF_C-MC_I.sav": Path("2023_11_models_length-agnostic") / "RF_C-MC_I.sav",
    "RF_C_MC_I.sav": Path("2023_11_models_length-agnostic") / "RF_C_MC_I.sav",
    "RF_I_P.sav": Path("2023_11_original_default_features-agnostic") / "RF_I_P.sav",
    "sam_vit_l_0b3195.pth": Path("sam_vit_l_0b3195.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download facet_ml model files from Zenodo and the Segment Anything "
            "checkpoint into facet_ml/static/Models."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=STATIC_FOLDER / "Models",
        help="Destination directory for model files.",
    )
    parser.add_argument(
        "--zenodo-record",
        default=ZENODO_RECORD_ID,
        help="Zenodo record id or DOI URL containing the model files.",
    )
    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip downloading the Segment Anything checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def extract_record_id(value: str) -> str:
    match = re.search(r"zenodo(?:\.org)?(?:/records/|/record/|\.)(\d+)", value)
    if match:
        return match.group(1)
    if value.isdigit():
        return value
    return value


def stream_download(url: str, destination: Path, overwrite: bool, timeout: float) -> None:
    if destination.exists() and not overwrite:
        print(f"skip {destination} (already exists)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with NamedTemporaryFile(delete=False, dir=destination.parent) as tmp_file:
            tmp_path = Path(tmp_file.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
    tmp_path.replace(destination)
    print(f"saved {destination}")


def zenodo_files(record_id: str, timeout: float) -> list[dict]:
    response = requests.get(
        f"{ZENODO_API_BASE}/{record_id}",
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    files = data.get("files", [])
    if not files:
        raise RuntimeError(f"No files were found in Zenodo record {record_id}.")
    return files


def destination_for(filename: str, models_dir: Path) -> Path:
    return models_dir / MODEL_DESTINATIONS.get(filename, Path(filename))


def download_zenodo_models(
    models_dir: Path, record_id: str, overwrite: bool, timeout: float
) -> None:
    for file_info in zenodo_files(record_id, timeout):
        key = file_info.get("key") or file_info.get("filename")
        if not key:
            continue
        link = file_info.get("links", {}).get("self")
        if not link:
            continue
        stream_download(
            link,
            destination_for(key, models_dir),
            overwrite=overwrite,
            timeout=timeout,
        )


def download_sam_model(
    models_dir: Path, overwrite: bool, timeout: float
) -> None:
    stream_download(
        SAM_URL,
        destination_for("sam_vit_l_0b3195.pth", models_dir),
        overwrite=overwrite,
        timeout=timeout,
    )


def report_missing(models_dir: Path) -> int:
    missing = []
    for name, model_path in STATIC_MODELS.items():
        expected = models_dir / Path(model_path).relative_to(STATIC_FOLDER / "Models")
        if not expected.exists():
            missing.append((name, expected))

    if not missing:
        print("all expected model files are present")
        return 0

    print("missing expected model files:")
    for name, path in missing:
        print(f"  {name}: {path}")
    return 1


def main() -> int:
    args = parse_args()
    record_id = extract_record_id(args.zenodo_record)
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_zenodo_models(
            models_dir=models_dir,
            record_id=record_id,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
        if not args.skip_sam:
            download_sam_model(
                models_dir=models_dir,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
    except requests.RequestException as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return report_missing(models_dir)


if __name__ == "__main__":
    raise SystemExit(main())
