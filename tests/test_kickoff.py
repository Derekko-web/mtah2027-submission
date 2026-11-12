"""D27 on synthetic data: reading kick-off tags from a SportsCode file and the dead-ball rule."""

from pathlib import Path

import numpy as np

from dfl_submission import kickoff


# Two synthetic halves, far apart and longer than the 5-minute limit (7,500 frames).
BLOCKS = [(1, 1000, 9999), (2, 12000, 29999)]
LENGTH = 31000


def instance(start, end, code="SET PIECES", labels=(("Type", "Kick Off (Start)"),)):
    tags = "".join(f"<label><group>{group}</group><text>{text}</text></label>" for group, text in labels)
    return f"<instance><ID>0</ID><start>{start}</start><end>{end}</end><code>{code}</code>{tags}</instance>"


def write_tags(folder, *instances):
    path = Path(folder) / "tags.xml"
    path.write_text("<file><ALL_INSTANCES>" + "".join(instances) + "</ALL_INSTANCES></file>", encoding="utf-8")
    return path


def live_match():
    """D15-like labels: a team owns every graded frame, X elsewhere."""
    labels = np.full(LENGTH, "X", dtype="<U1")
    for _, first, last in BLOCKS:
        labels[first:last + 1] = "H"
    return labels


def frames(seconds):
    return int(round(seconds * kickoff.FPS))


def test_read_tags_keeps_only_kick_off_start_set_pieces(tmp_path):
    path = write_tags(tmp_path,
                      instance(36.0, 41.0),
                      instance(100.0, 104.0, labels=(("Type", "Kick Off (After Goal)"),)),
                      instance(200.0, 204.0, labels=(("Type", "Throw-in"),)),
                      instance(300.0, 304.0, code="BUILD UP"),
                      instance(400.0, 404.0, labels=(("Half", "2nd Half"),)),
                      instance(500.01, 505.0, labels=(("Half", "2nd Half"), ("Type", "Kick Off (Start)"))))
    assert kickoff.read_tags(path) == [(900, 1025), (12500, 12625)]


def test_read_tags_matches_code_group_and_type_ignoring_case_and_spaces(tmp_path):
    path = write_tags(tmp_path,
                      instance(480.0, 485.0, code=" set pieces ", labels=(("TYPE", " kick off (START) "),)))
    assert kickoff.read_tags(path) == [(12000, 12125)]


def test_second_half_is_dead_up_to_the_tag_start():
    labels = live_match()
    start = 12000 + 2934
    result = kickoff.apply(labels, [(start, start + 125)], BLOCKS)
    assert (result[12000:start] == "D").all()
    assert (result[start:30000] == "H").all()
    assert np.array_equal(result[:12000], labels[:12000])
    assert (result[30000:] == "X").all()


def test_input_labels_are_not_modified():
    labels = live_match()
    kept = labels.copy()
    kickoff.apply(labels, [(12500, 12625)], BLOCKS)
    assert np.array_equal(labels, kept)


def test_tag_starting_before_the_half_changes_nothing():
    labels = live_match()
    assert np.array_equal(kickoff.apply(labels, [(900, 1025)], BLOCKS), labels)
    assert np.array_equal(kickoff.apply(labels, [(12000, 12125)], BLOCKS), labels)


def test_first_half_tag_and_second_half_tag_together():
    labels = live_match()
    result = kickoff.apply(labels, [(900, 1025), (12500, 12625)], BLOCKS)
    assert np.array_equal(result[:12000], labels[:12000])
    assert (result[12000:12500] == "D").all() and (result[12500:30000] == "H").all()


def test_no_tag_changes_nothing():
    labels = live_match()
    assert np.array_equal(kickoff.apply(labels, [], BLOCKS), labels)


def test_two_tags_in_one_half_change_nothing():
    labels = live_match()
    assert np.array_equal(kickoff.apply(labels, [(12500, 12625), (13000, 13125)], BLOCKS), labels)


def test_tag_beyond_the_limit_changes_nothing():
    labels = live_match()
    late = 12000 + kickoff.LIMIT + 1
    assert np.array_equal(kickoff.apply(labels, [(late, late + 125)], BLOCKS), labels)


def test_tag_at_the_limit_is_used():
    labels = live_match()
    edge = 12000 + kickoff.LIMIT
    result = kickoff.apply(labels, [(edge, edge + 125)], BLOCKS)
    assert (result[12000:edge] == "D").all() and result[edge] == "H"


def test_tag_ending_before_the_half_does_not_belong_to_it():
    labels = live_match()
    # Ends one frame before the second half: it belongs to no half, so neither half changes.
    assert np.array_equal(kickoff.apply(labels, [(11800, 11999)], BLOCKS), labels)
    # Two tags, but only one ends inside the half's reach: that one is used.
    result = kickoff.apply(labels, [(11800, 11999), (12500, 12625)], BLOCKS)
    assert (result[12000:12500] == "D").all() and result[12500] == "H"


def test_other_codes_and_types_change_nothing(tmp_path):
    labels = live_match()
    path = write_tags(tmp_path,
                      instance(500.0, 505.0, labels=(("Type", "Kick Off (After Goal)"),)),
                      instance(500.0, 505.0, code="BUILD UP", labels=(("Type", "Kick Off (Start)"),)))
    assert np.array_equal(kickoff.constrained(labels, path, BLOCKS), labels)


def test_constrained_reads_the_file_and_applies_the_rule(tmp_path):
    labels = live_match()
    path = write_tags(tmp_path, instance(36.0, 41.0), instance(500.0, 505.0, code="Set Pieces",
                                                               labels=(("type", "KICK OFF (START)"),)))
    result = kickoff.constrained(labels, path, BLOCKS)
    assert np.array_equal(result[:12000], labels[:12000])
    assert (result[12000:12500] == "D").all() and (result[12500:30000] == "H").all()
    assert (result[30000:] == "X").all()


def test_frames_outside_the_halves_stay_x():
    labels = live_match()
    result = kickoff.apply(labels, [(12500, 12625)], BLOCKS)
    outside = np.ones(LENGTH, dtype=bool)
    for _, first, last in BLOCKS:
        outside[first:last + 1] = False
    assert (result[outside] == "X").all()


def test_build_writes_d15_labels_after_the_rule(tmp_path):
    labels = live_match()
    path = write_tags(tmp_path, instance(500.0, 505.0))
    written = {}
    originals = kickoff.d15_labels, kickoff.tag_path, kickoff.half_blocks, kickoff.write_csv
    kickoff.d15_labels = lambda prepare=None: labels.copy()
    kickoff.tag_path = lambda: path
    kickoff.half_blocks = lambda: BLOCKS
    kickoff.write_csv = lambda output, values: written.update(output=output, labels=values)
    try:
        kickoff.build(tmp_path / "out.csv")
    finally:
        kickoff.d15_labels, kickoff.tag_path, kickoff.half_blocks, kickoff.write_csv = originals
    assert written["output"] == tmp_path / "out.csv"
    assert np.array_equal(written["labels"], kickoff.apply(labels, [(12500, 12625)], BLOCKS))


def test_the_limit_is_five_minutes():
    assert kickoff.LIMIT == 5 * 60 * 25 and kickoff.FPS == 25
