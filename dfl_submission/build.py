"""Generate a competition CSV with frozen DFL models or the raw starter baseline."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, GAMES, half_video_blocks, source_frames
from starter import baseline


def dfl_labels(folder):
    from dfl_submission.decoder import decode
    from dfl_submission.models import MODEL_FILES, load_models, predict
    from dfl_transfer_data import digest

    report = json.loads((folder / "external_validation.json").read_text(encoding="utf-8"))
    if report["holdout_passed"] is not True:
        raise ValueError("The DFL recipe did not pass external validation. Use --raw.")
    frozen = json.loads((folder / "recipe.json").read_text(encoding="utf-8"))
    for name in MODEL_FILES:
        if digest(folder / name) != frozen["models"][name]:
            raise ValueError(f"Checkpoint identity mismatch: {name}")
    models = load_models(folder)

    def decode_half(positions, ball, teams, pitch):
        home, dead, _ = predict(*models, positions, ball, teams)
        return decode(home, dead, **frozen["recipe"])

    return tactical_halves(decode_half)


def tactical_halves(decode_half, *, centered=False):
    from broadcast_alignment import load_source

    source = load_source("tactical")
    positions, ball = source["positions"], source["ball"]
    positions[~source["valid"]] = np.nan
    pitch = np.array(GAMES[GAME]["pitch"])
    ball_valid = (np.isfinite(ball).all(axis=1) & (ball >= -5).all(axis=1)
                  & (ball <= pitch + 5).all(axis=1) & source["valid"])
    ball[~ball_valid] = np.nan
    if centered:
        positions, ball = positions - pitch / 2, ball - pitch / 2

    labels = np.full(source_frames(), "X", dtype="<U1")
    classes = np.array(pm.LABELS)
    for half, first, last in half_video_blocks(GAME):
        span = slice(first, last + 1)
        print(f"Predicting tactical half {half}", flush=True)
        labels[span] = classes[decode_half(positions[span], ball[span], source["teams"], pitch)]
    return labels


def write_csv(output, labels):
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                            prefix=f".{output.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write("frame,label\n")
            handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
            handle.close()
            reason = pm.validate_submission(pm.read_label_csv(temporary), source_frames())
            if reason:
                raise ValueError(reason)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Validated {len(labels):,} rows in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).with_name("models"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw", action="store_true", help="Use the unchanged, unsmoothed starter baseline")
    args = parser.parse_args()
    labels = baseline.smooth(baseline.build(), 0) if args.raw else dfl_labels(args.model_dir)
    write_csv(args.out, labels)
