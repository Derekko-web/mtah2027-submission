"""Static checks that the submission code cannot read the public ground truth."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = sorted([*ROOT.glob("dfl_submission/*.py"), *ROOT.glob("experiments/*.py"), ROOT / "games.py",
               *ROOT.glob("starter/*.py")])
GROUND_TRUTH = re.compile(r"evaluation/gt|possession_gt|\bgt_path\b|public_gt|read_ground_truth")
GRADED_FRAMES = re.compile(r"\b(11185|90651|90652|110808|113742|190197|190198)\b")


def test_build_code_never_names_the_ground_truth():
    offenders = [str(path.relative_to(ROOT)) for path in CODE if GROUND_TRUTH.search(path.read_text())]
    assert not offenders, offenders


def test_no_frame_literals_outside_games_module():
    offenders = [str(path.relative_to(ROOT)) for path in CODE
                 if path.name != "games.py" and GRADED_FRAMES.search(path.read_text())]
    assert not offenders, offenders


def test_ground_truth_folder_is_not_shipped():
    assert not (ROOT / "evaluation" / "gt").exists()
