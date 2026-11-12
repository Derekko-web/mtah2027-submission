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


# -- D29: dead ball from a tagged goal to the kick-off after it ------------------------------------

def goal(start, end):
    return instance(start, end, code="GOALS", labels=(("Team", "Barcelona"),))


def restart(start, end):
    return instance(start, end, labels=(("Type", "Kick Off (After Goal)"),))


def test_read_instances_keeps_code_type_and_frames(tmp_path):
    path = write_tags(tmp_path, goal(100.0, 108.0), restart(160.0, 165.0),
                      instance(200.0, 204.0, code=" Build Up ", labels=(("TYPE", "x"), ("type", " Throw-In "))))
    assert kickoff.read_instances(path) == [
        {"code": "goals", "type": None, "start": 2500, "end": 2700},
        {"code": "set pieces", "type": "kick off (after goal)", "start": 4000, "end": 4125},
        {"code": "build up", "type": "throw-in", "start": 5000, "end": 5100}]


def test_read_tags_is_unchanged_by_the_instance_reader(tmp_path):
    path = write_tags(tmp_path, instance(36.0, 41.0), restart(160.0, 165.0), goal(100.0, 108.0))
    assert kickoff.read_tags(path) == [(900, 1025)]


def spans(*items):
    return [{"code": code, "type": kind, "start": start, "end": end} for code, kind, start, end in items]


GOAL, AFTER = ("goals", None), ("set pieces", "kick off (after goal)")


def test_goal_end_to_kick_off_start_becomes_dead():
    tags = spans((*GOAL, 3000, 3200), (*AFTER, 5000, 5125))
    assert kickoff.goal_spans(tags, BLOCKS) == [(3200, 5000)]
    result = kickoff.apply_goals(live_match(), tags, BLOCKS)
    assert (result[3200:5000] == "D").all() and (result[1000:3200] == "H").all() and (result[5000:10000] == "H").all()
    assert (result[:1000] == "X").all()


def test_latest_goal_before_the_kick_off_is_used():
    tags = spans((*GOAL, 2000, 2100), (*GOAL, 3000, 3200), (*AFTER, 5000, 5125))
    assert kickoff.goal_spans(tags, BLOCKS) == [(3200, 5000)]


def test_three_minutes_is_the_reach():
    assert kickoff.goal_spans(spans((*GOAL, 3000, 3500), (*AFTER, 8000, 8125)), BLOCKS) == [(3500, 8000)]
    assert kickoff.goal_spans(spans((*GOAL, 3000, 3499), (*AFTER, 8000, 8125)), BLOCKS) == []
    assert kickoff.GOAL_REACH == 3 * 60 * 25


def test_goal_in_the_other_half_is_not_a_pair():
    tags = spans((*GOAL, 9000, 9900), (*AFTER, 12100, 12200))
    assert kickoff.goal_spans(tags, BLOCKS) == []


def test_goal_ending_after_the_kick_off_start_is_not_a_pair():
    assert kickoff.goal_spans(spans((*GOAL, 4000, 5050), (*AFTER, 5000, 5125)), BLOCKS) == []


def test_two_goals_sharing_the_latest_end_change_nothing():
    tags = spans((*GOAL, 3000, 3200), (*GOAL, 3100, 3200), (*AFTER, 5000, 5125))
    assert kickoff.goal_spans(tags, BLOCKS) == []


def test_one_goal_paired_with_two_kick_offs_changes_nothing():
    tags = spans((*GOAL, 3000, 3200), (*AFTER, 4000, 4125), (*AFTER, 5000, 5125))
    assert kickoff.goal_spans(tags, BLOCKS) == []


def test_another_goal_starting_inside_the_stretch_changes_nothing():
    tags = spans((*GOAL, 3000, 3200), (*GOAL, 4500, 5100), (*AFTER, 5000, 5125))
    assert kickoff.goal_spans(tags, BLOCKS) == []


def test_kick_off_start_and_other_codes_are_not_kick_offs_after_goals():
    tags = spans((*GOAL, 3000, 3200), ("set pieces", "kick off (start)", 5000, 5125), ("goals", "x", 4000, 4100),
                 ("build up", "kick off (after goal)", 5000, 5125))
    assert kickoff.goal_spans(tags, BLOCKS) == []


def test_each_goal_and_its_kick_off_pair_independently():
    tags = spans((*GOAL, 3000, 3200), (*AFTER, 5000, 5125), (*GOAL, 20000, 20300), (*AFTER, 22000, 22100))
    assert kickoff.goal_spans(tags, BLOCKS) == [(3200, 5000), (20300, 22000)]
    result = kickoff.apply_goals(live_match(), tags, BLOCKS)
    assert int((result == "D").sum()) == (5000 - 3200) + (22000 - 20300)


def test_goal_rule_leaves_its_input_unchanged():
    labels = live_match()
    kept = labels.copy()
    kickoff.apply_goals(labels, spans((*GOAL, 3000, 3200), (*AFTER, 5000, 5125)), BLOCKS)
    assert np.array_equal(labels, kept)


def test_model_labels_prepare_the_ball_before_the_model_sees_it():
    seen = []

    def fake_halves(decode_half, *, centered, pitch, merge_ball=None):
        assert centered is True and pitch == (105.0, 68.0) and merge_ball is None
        return decode_half(np.zeros((3, 2, 2)), np.array([[1., 2.], [np.nan, np.nan], [3., 4.]]), np.array([0, 1]), pitch)

    def fake_signals(teacher, positions, ball, teams, pitch):
        seen.append((teacher, ball.copy()))
        return None, np.array([0, 1, 2])

    originals = kickoff.tactical_halves, kickoff.diagnosis.half_signals
    kickoff.tactical_halves, kickoff.diagnosis.half_signals = fake_halves, fake_signals
    try:
        labels = kickoff.model_labels("model", (105.0, 68.0), prepare=lambda ball: np.nan_to_num(ball))
    finally:
        kickoff.tactical_halves, kickoff.diagnosis.half_signals = originals
    assert list(labels) == [0, 1, 2]
    assert seen[0][0] == "model" and np.array_equal(seen[0][1], [[1., 2.], [0., 0.], [3., 4.]])


def test_model_labels_pass_a_ball_merge_to_the_tracking():
    hook = object()
    seen = {}

    def fake_halves(decode_half, *, centered, pitch, merge_ball=None):
        seen["merge_ball"] = merge_ball
        return np.zeros(3)

    original = kickoff.tactical_halves
    kickoff.tactical_halves = fake_halves
    try:
        kickoff.model_labels("model", (105.0, 68.0), merge_ball=hook)
    finally:
        kickoff.tactical_halves = original
    assert seen["merge_ball"] is hook
