"""Round 8's build: D15 with D27's kick-off rule, plus each round 8 step whose switch is on (DECISIONS.md C3).

With every switch off this is exactly `kickoff build` (D27), so the file C1 checked is reproduced
byte for byte. `--metrica` uses D30's confirmed final fit, `--broadcast` merges D31's broadcast ball
before the models, `--adapt` applies D32's confirmed per-match adaptation (only with written
organiser approval), and `--goals` applies D29 after D27. Each step checks its own records.
"""

import argparse
from pathlib import Path

from dfl_submission import broadcast_trial, kickoff, metrica_trial
from dfl_submission.build import write_csv


SWITCHES = ("metrica", "broadcast", "adapt", "goals")


def d15_model():
    from dfl_submission import metrica_sample as ms
    return ms.reference_model("D15")


def labels(*, metrica=False, broadcast=False, adapt=False, goals=False):
    from dfl_submission import zoo
    if not (metrica or broadcast or adapt):
        decoded = kickoff.d15_labels()
    else:
        _, frozen = zoo.d15_setting()
        model = metrica_trial.confirmed_teacher() if metrica else d15_model()
        merge_ball = broadcast_trial.confirmed_merger() if broadcast else None
        if adapt:
            from dfl_submission import vendor_adaptation
            decoded = vendor_adaptation.competition_labels(model, frozen["pitch"], merge_ball)
        else:
            decoded = kickoff.model_labels(model, frozen["pitch"], merge_ball=merge_ball)
    path, blocks = kickoff.tag_path(), kickoff.half_blocks()
    result = kickoff.constrained(decoded, path, blocks)
    if goals:
        result = kickoff.apply_goals(result, kickoff.read_instances(path), blocks)
    return result


def build(output, **switches):
    write_csv(output, labels(**switches))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    for name in SWITCHES:
        finished.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    build(args.out, **{name: getattr(args, name) for name in SWITCHES})


if __name__ == "__main__":
    main()
