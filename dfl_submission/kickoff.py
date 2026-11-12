"""D27: dead ball until the tagged kick-off (DECISIONS.md D27).

After D15 decodes the match, every frame from a half's first graded frame up to the start of that
half's `Kick Off (Start)` Smart Tagging tag becomes `D`: no team has possession before the kick-off.
A tag belongs to a half if it ends at or after the half's first frame and starts at most 5 minutes
after it; a half with no such tag, or more than one, is untouched. The organisers' loader
(`starter/load_smart_tagging.py`) says each kick-off tag is a window of about 5 s containing the real
kick-off, so its start is never later than the kick-off and the rule cannot overwrite live play.

`read_tags` reads the SportsCode XML with the standard library, as the organisers' loader does.
`apply` is the rule on a full-length label array. `build` writes the competition file: D15
(`scale_trial`'s build path), then the rule. Nothing here is fitted, tuned or scored.

D29 (DECISIONS.md D29): `read_instances` reads every instance's code, type and frames, `goal_spans`
pairs each `Kick Off (After Goal)` tag with the latest `GOALS` tag that ends at most 3 minutes before
it in the same half, and `apply_goals` makes the stretch between them dead. `model_labels` is the
build path for any model with D15's structure.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.build import tactical_halves, write_csv


FPS = 25                               # DATA.md: video_frame = round(seconds * 25)
LIMIT = 5 * 60 * FPS                   # a tag starting more than 5 minutes into a half is not its kick-off
CODE = "set pieces"                    # compared after trimming and lower-casing, as DATA.md advises
TYPE_GROUP = "type"
KICK_OFF = "kick off (start)"          # not "kick off (after goal)"
DEAD = "D"
GOAL_CODE = "goals"                    # D29
AFTER_GOAL = "kick off (after goal)"
GOAL_REACH = 3 * 60 * FPS              # a goal tag ending more than 3 minutes before the kick-off is not its goal


def normal(text):
    return (text or "").strip().lower()


def read_tags(path, fps=FPS):
    """[(start_frame, end_frame)] of every `SET PIECES` / `Kick Off (Start)` instance, in file order.

    Frames are rounded to the nearest frame as the organisers' loader rounds them. When a label
    group repeats, the last value wins, as in that loader.
    """
    tags = []
    for instance in ET.parse(path).iter("instance"):
        if normal(instance.findtext("code")) != CODE:
            continue
        kind = None
        for label in instance.findall("label"):
            if normal(label.findtext("group")) == TYPE_GROUP:
                kind = normal(label.findtext("text"))
        if kind != KICK_OFF:
            continue
        tags.append((int(round(float(instance.findtext("start")) * fps)),
                     int(round(float(instance.findtext("end")) * fps))))
    return tags


def read_instances(path, fps=FPS):
    """[{code, type, start, end}] of every instance, in file order (D29).

    Code and type are trimmed and lower-cased; type is the last `Type` label, or None. Frames are
    rounded as `read_tags` rounds them.
    """
    instances = []
    for instance in ET.parse(path).iter("instance"):
        kind = None
        for label in instance.findall("label"):
            if normal(label.findtext("group")) == TYPE_GROUP:
                kind = normal(label.findtext("text"))
        instances.append({"code": normal(instance.findtext("code")), "type": kind,
                          "start": int(round(float(instance.findtext("start")) * fps)),
                          "end": int(round(float(instance.findtext("end")) * fps))})
    return instances


def half_of(frame, blocks):
    for half, first, last in blocks:
        if first <= frame <= last:
            return half
    return None


def goal_spans(instances, blocks):
    """[(first dead frame, kick-off tag start)] for every unambiguous goal and kick-off pair (D29)."""
    goals = [tag for tag in instances if tag["code"] == GOAL_CODE]
    restarts = [tag for tag in instances if tag["code"] == CODE and tag["type"] == AFTER_GOAL]
    pairs = []
    for restart in restarts:
        half = half_of(restart["start"], blocks)
        if half is None:
            continue
        eligible = [index for index, tag in enumerate(goals) if half_of(tag["start"], blocks) == half
                    and tag["end"] <= restart["start"] and restart["start"] - tag["end"] <= GOAL_REACH]
        if not eligible:
            continue
        latest = max(goals[index]["end"] for index in eligible)
        chosen = [index for index in eligible if goals[index]["end"] == latest]
        if len(chosen) != 1:
            continue
        end = goals[chosen[0]]["end"]
        if any(index != chosen[0] and end < tag["start"] < restart["start"] for index, tag in enumerate(goals)):
            continue
        pairs.append((chosen[0], end, restart["start"]))
    used = [index for index, _, _ in pairs]
    return sorted((end, start) for index, end, start in pairs if used.count(index) == 1 and end < start)


def apply_goals(labels, instances, blocks):
    """A copy of full-length `labels` with every goal-to-kick-off stretch dead (D29)."""
    result = np.array(labels, copy=True)
    for first, stop in goal_spans(instances, blocks):
        result[first:stop] = DEAD
    return result


def apply(labels, tags, blocks):
    """A copy of full-length `labels` with each half dead from its first frame to its kick-off tag's start.

    `blocks` is [(half, first_frame, last_frame)], as `games.half_video_blocks` gives them.
    """
    result = np.array(labels, copy=True)
    for _, first, last in blocks:
        owned = [start for start, end in tags if end >= first and start <= first + LIMIT]
        if len(owned) != 1:
            continue
        stop = min(owned[0], last + 1)
        if stop > first:
            result[first:stop] = DEAD
    return result


def constrained(labels, path, blocks):
    """The rule with the tags read from the SportsCode file at `path`."""
    return apply(labels, read_tags(path), blocks)


# -- the competition build -------------------------------------------------------------------------

def tag_path():
    from games import require
    return require(["smart_tagging"])[0]


def half_blocks():
    from games import GAME, half_video_blocks
    return half_video_blocks(GAME)


def model_labels(model, pitch, prepare=None, merge_ball=None):
    """A model's labels for the competition match, decoded as `scale_trial.build` decodes D15's.

    `prepare(ball)`, if given, changes each half's ball before the models see it (D28's fill).
    `merge_ball`, if given, fills missing balls of the whole match first (D31).
    """
    def decode_half(positions, ball, teams, pitch_):
        if prepare is not None:
            ball = prepare(ball)
        return diagnosis.half_signals(model, positions, ball, teams, pitch_)[1]

    return tactical_halves(decode_half, centered=True, pitch=pitch, merge_ball=merge_ball)


def d15_labels(prepare=None):
    """D15's labels for the competition match, built exactly as `scale_trial.build` builds them."""
    from dfl_submission import scale_trial
    if json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The D15 networks did not pass confirmation; build D3 instead")
    frozen, setting = scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)
    reference, packages, d3 = scale_trial.teacher(setting)
    model = (reference, packages, dict(d3, recipe=frozen["recipe"]))
    return model_labels(model, frozen["pitch"], prepare)


def build(output):
    """The competition file: D15, then D27's rule."""
    write_csv(output, constrained(d15_labels(), tag_path(), half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
