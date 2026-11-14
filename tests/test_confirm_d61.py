"""D61's set D check, rule and half comparison on synthetic fixtures (and the tracked PFF source list)."""

import numpy as np

from dfl_submission import confirm_d61 as cd, diagnosis


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def test_set_d_is_the_first_and_second_train_slots_of_d7_s_split():
    check = cd.check_set_d(cd.pinned_ids())
    assert check["set_d_is_slots_1_and_2"] and check["set_e_is_slot_3"] and check["slots_are_pff_split_train"]
    assert len(cd.SET_D) == 16 and len(set(cd.SET_D)) == 16 and not set(cd.SET_D) & set(cd.SET_E)
    slots = cd.train_slots(range(64))
    assert slots == {1: tuple(range(4, 64, 8)), 2: tuple(range(5, 64, 8)), 3: tuple(range(6, 64, 8))}


def change(pff=.012, primary=0.0, long_hidden=0.0, conventions=0.0, guards=0.0):
    return {"pff": pff, "primary": primary, "long_hidden": long_hidden,
            "conventions": {"aba1.75": conventions, "aba4": 0.0, "aba4.5": 0.0},
            "guards": {"dead_carried": guards, "team_swap": 0.0}}


def test_rule_applies_d61_s_conditions():
    assert all(cd.rule(change()).values())
    assert all(cd.rule(change(pff=.01)).values())                          # exactly +0.01 passes
    assert not cd.rule(change(pff=.0099))["set_d"]
    assert cd.rule(change(primary=-.01))["primary"] and not cd.rule(change(primary=-.0101))["primary"]
    assert not cd.rule(change(long_hidden=-.0101))["long_hidden"]
    assert not cd.rule(change(conventions=-.0101))["conventions"]
    assert not cd.rule(change(guards=-.0101))["guards"]


def half(index, value, view="observed"):
    signals = tuple(np.full(10, value + k, dtype=np.float32) for k in range(4))
    return diagnosis.HalfSignals("dfl", "WQQ", view, index, np.zeros(10, dtype=np.int8), signals, np.zeros(10, dtype=np.int8))


def test_same_halves_compares_bit_for_bit_by_key():
    fresh = [half(0, 1.0), half(1, 2.0)]
    assert all(cd.same_halves(fresh, [half(1, 2.0), half(0, 1.0)]).values())
    changed = [half(0, 1.0), half(1, 2.0 + 1e-6)]
    result = cd.same_halves(fresh, changed)
    assert result["dfl/WQQ/observed/0"] and not result["dfl/WQQ/observed/1"]
    assert not cd.same_halves(fresh, [half(0, 1.0)])["dfl/WQQ/observed/1"]   # a missing half does not match


def test_records_once(tmp_path):
    cd.write_new(tmp_path / "r.json", {"a": 1})
    assert refused(lambda: cd.write_new(tmp_path / "r.json", {}), FileExistsError)
    assert cd.COST == 14 and cd.GAIN == .01 and cd.LOSS == .01


def fake_parsed(tmp_path, games):
    import json
    folder = tmp_path / "parsed"
    folder.mkdir(parents=True)
    manifest = {"sources_sha256": "x", "parsed": {}}
    for game in games:
        for index in (1, 2):
            (folder / f"{game}_{index}.npz").write_bytes(b"x")
            manifest["parsed"][f"{game}_{index}.npz"] = [1, f"{game}{index}"]
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_parse_set_d_leaves_out_refused_games_and_needs_twelve(tmp_path):
    import json
    first = dict(cd.FIRST_ATTEMPT)
    folder = fake_parsed(tmp_path, first["parsed"])
    refuse = {3840, 10515}

    def preparer(folder):
        def prepare_game(game):
            if game in refuse:
                raise ValueError(f"Repeated PFF frame 1 in {game}")
            manifest = json.loads((folder / "manifest.json").read_text())
            for index in (1, 2):
                (folder / f"{game}_{index}.npz").write_bytes(b"y")
                manifest["parsed"][f"{game}_{index}.npz"] = [1, f"{game}{index}"]
            (folder / "manifest.json").write_text(json.dumps(manifest))
        return prepare_game

    note = cd.parse_set_d(folder, None, {}, prepare_game=preparer(folder), first_attempt=first, output=tmp_path / "n.json")
    assert len(note["games"]) == 14 and 3840 not in note["games"] and 10515 not in note["games"]
    assert set(note["left_out"]) == {"3840", "10515"} and note["left_out"]["3840"].startswith("ValueError: Repeated")
    assert sorted(note["games"] + [3840, 10515]) == sorted(cd.SET_D)
    assert refused(lambda: cd.write_new(tmp_path / "n.json", {}), FileExistsError)       # the parse note is written once
    # a game already parsed outside the first attempt stops the step
    folder2 = fake_parsed(tmp_path / "b", first["parsed"] + [3841])
    assert refused(lambda: cd.parse_set_d(folder2, None, {}, prepare_game=preparer(folder2), first_attempt=first,
                                          output=tmp_path / "b" / "n.json"), ValueError)
    # fewer than twelve readable games stop D61 unscored
    folder3 = fake_parsed(tmp_path / "c", first["parsed"])
    refuse.update({3841, 3848, 3849, 3856})
    assert refused(lambda: cd.parse_set_d(folder3, None, {}, prepare_game=preparer(folder3), first_attempt=first,
                                          output=tmp_path / "c" / "n.json"), SystemExit)


def row(half, sign, used=None, agrees=True, fallback=None, group="pff_set_d"):
    return {"half": half, "group": group, "d": None if sign is None else 3.0 * sign, "frames": 100, "sign": sign,
            "sign_used": sign if used is None else used, "first_frame": 5,
            "first_sign": None if sign is None else (sign if agrees else -sign),
            "agrees": agrees, "fallback": fallback, "max_visible": 11, "max_visible_all_frames": 11, "first_difference": 1.0}


def test_sign_report_reports_first_frames_and_checks_opposite_halves():
    original = cd.st.summarise
    cd.st.summarise = lambda rows, training: {"halves": len(rows)}
    try:
        table = {"a": row("pff/3841/observed/0", 1), "b": row("pff/3841/observed/1", -1, agrees=False),
                 "c": row("pff/3816/observed/0", -1), "d": row("pff/3816/observed/1", None, used=1,
                                                               fallback={"other_half": "x", "other_half_sign": -1})}
        report = cd.sign_report(table)
        assert report["first_frame_disagrees"] == ["pff/3841/observed/1"]
        assert report["fallbacks"] == ["pff/3816/observed/1"]
        assert all(report["opposite_halves"].values()) and set(report["opposite_halves"]) == {"pff/3841/observed",
                                                                                              "pff/3816/observed"}
        assert report["per_half"]["pff/3841/observed/1"]["sign_used"] == -1
        assert report["per_half"]["pff/3841/observed/1"]["first_sign"] == 1
        same = dict(table, b=row("pff/3841/observed/1", 1))
        assert not cd.sign_report(same)["opposite_halves"]["pff/3841/observed"]
        lone = {"a": row("pff/3841/observed/0", 1)}
        assert not cd.sign_report(lone)["opposite_halves"]["pff/3841/observed"]
    finally:
        cd.st.summarise = original
