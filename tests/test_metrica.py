"""Synthetic checks of the Metrica sample-match reader, targets and veto (DECISIONS.md D19).

No Metrica data is committed; the fixtures mimic its CSV, EPTS and JSON layouts.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfl_submission import metrica_sample as ms
from evaluation import possession_metrics as pm


H, A, D, X = pm.H, pm.A, pm.D, pm.X


def event(index, team, kind, start, end=None, subtypes=(), period=1):
    return ms.Event(index, team, kind, tuple(subtypes), period, start, start if end is None else end)


# -- tracking --------------------------------------------------------------------------------------

def test_to_metres_scales_and_centres_without_flipping():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.25, 0.75], [1.1, 0.5], [np.nan, 0.5]])
    metres = ms.to_metres(points, (105.0, 68.0))
    np.testing.assert_allclose(metres[:4], [[-52.5, -34.0], [52.5, 34.0], [0.0, 0.0], [-26.25, 17.0]], rtol=1e-6)
    assert np.isnan(metres[4]).all()  # 10.5 m beyond the goal line, past the 5 m limit
    assert np.isnan(metres[5]).all() and metres.dtype == np.float32


def write_csv_side(path, team, numbers, rows):
    lines = [",,," + "".join(f"{team},," for _ in numbers) + ",",
             ",,," + "".join(f"{number},," for number in numbers) + ",",
             "Period,Frame,Time [s]," + "".join(f"Player{number},," for number in numbers) + "Ball,"]
    for period, frame, players, ball in rows:
        values = [str(period), str(frame), f"{frame / 25:.2f}"]
        values += [str(value) for point in players for value in point] + [str(value) for value in ball]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n")


def test_csv_tracking_reads_both_teams_and_checks_the_ball():
    rows_home = [(1, 1, [(0.1, 0.2), ("NaN", "NaN")], (0.5, 0.5)), (2, 2, [(0.3, 0.4), (0.6, 0.7)], ("NaN", "NaN"))]
    rows_away = [(1, 1, [(0.9, 0.8)], (0.5, 0.5)), (2, 2, [(0.7, 0.6)], ("NaN", "NaN"))]
    with TemporaryDirectory() as folder:
        home, away = Path(folder) / "home.csv", Path(folder) / "away.csv"
        write_csv_side(home, "Home", (11, 1), rows_home)
        write_csv_side(away, "Away", (25,), rows_away)
        frames, periods, positions, ball, teams = ms.csv_tracking(home, away)
        np.testing.assert_array_equal(frames, [1, 2])
        np.testing.assert_array_equal(periods, [1, 2])
        np.testing.assert_array_equal(teams, [0, 0, 1])
        np.testing.assert_allclose(positions[0], [[0.1, 0.2], [np.nan, np.nan], [0.9, 0.8]])
        np.testing.assert_allclose(positions[1], [[0.3, 0.4], [0.6, 0.7], [0.7, 0.6]])
        assert np.isfinite(ball[0]).all() and np.isnan(ball[1]).all()
        write_csv_side(away, "Away", (25,), [(1, 1, [(0.9, 0.8)], (0.4, 0.5)), rows_away[1]])
        try:
            ms.csv_tracking(home, away)
        except ValueError as error:
            assert "ball" in str(error)
        else:
            raise AssertionError("A ball that differs between the two files must be refused")


METADATA = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FrameRate>25</FrameRate>
      <ProviderGlobalParameters>
        <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
        <ProviderParameter><Name>first_half_end</Name><Value>3</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_start</Name><Value>4</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_end</Name><Value>5</Value></ProviderParameter>
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions><Session id="s"><MatchParameters><Score idLocalTeam="TA" idVisitingTeam="TB"/></MatchParameters>
      <FieldSize><Width>100</Width><Height>60</Height></FieldSize></Session></Sessions>
    <Teams><Team id="TA"><Name>Team A</Name></Team><Team id="TB"><Name>Team B</Name></Team></Teams>
    <Players>
      <Player id="P1" teamId="TA"/><Player id="P2" teamId="TB"/><Player id="P3" teamId="TA"/>
    </Players>
    <Devices><Device id="d"><Sensors><Sensor id="position"><Channels>
      <Channel id="x"><Unit>normalized</Unit></Channel><Channel id="y"><Unit>normalized</Unit></Channel>
    </Channels></Sensor></Sensors></Device></Devices>
    <PlayerChannels>
      <PlayerChannel channelId="x" id="p1_x" playerId="P1"/><PlayerChannel channelId="y" id="p1_y" playerId="P1"/>
      <PlayerChannel channelId="x" id="p2_x" playerId="P2"/><PlayerChannel channelId="y" id="p2_y" playerId="P2"/>
      <PlayerChannel channelId="x" id="p3_x" playerId="P3"/><PlayerChannel channelId="y" id="p3_y" playerId="P3"/>
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="3" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/></SplitRegister>
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/></SplitRegister>
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
    </DataFormatSpecification>
    <DataFormatSpecification endFrame="5" separator=":" startFrame="4">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/></SplitRegister>
        <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/></SplitRegister>
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
"""


def test_epts_follows_each_player_layout():
    lines = ["1:0.1,0.2;0.3,0.4:0.5,0.5\n", "2:0.1,0.2;NaN,NaN:NaN,NaN\n", "3:0.1,0.2;0.3,0.4:0.5,0.6\n",
             "4:0.7,0.8;0.3,0.4:0.5,0.5\n", "5:0.7,0.8;0.3,0.4:0.5,0.5\n"]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "metadata.xml"
        path.write_text(METADATA)
        metadata = ms.epts_metadata(path)
    assert metadata["pitch"] == (100.0, 60.0) and metadata["team_ids"] == ["TA", "TB"]
    assert metadata["periods"] == {1: (1, 3), 2: (4, 5)}
    frames, periods, positions, ball, teams = ms.epts_tracking(metadata, lines)
    np.testing.assert_array_equal(frames, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(periods, [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(teams, [0, 1, 0])
    np.testing.assert_allclose(positions[0], [[0.1, 0.2], [0.3, 0.4], [np.nan, np.nan]])
    np.testing.assert_allclose(positions[3], [[np.nan, np.nan], [0.3, 0.4], [0.7, 0.8]])  # P3 replaced P1
    assert np.isnan(positions[1, 1]).all() and np.isnan(ball[1]).all()
    np.testing.assert_allclose(ball[2], [0.5, 0.6])


def test_events_from_json_and_csv():
    items = {"data": [
        {"team": {"id": "TB"}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "KICK OFF"}, "period": 1,
         "start": {"frame": 10}, "end": {"frame": 10}},
        {"team": {"id": "TA"}, "type": {"name": "BALL LOST"}, "subtypes": [{"name": "HEAD"}, {"name": "INTERCEPTION"}],
         "period": 2, "start": {"frame": 20}, "end": {"frame": None}},
        {"team": {"id": "TA"}, "type": {"name": "PASS"}, "subtypes": None, "period": 2,
         "start": {"frame": 30}, "end": {"frame": 41}}]}
    with TemporaryDirectory() as folder:
        path = Path(folder) / "events.json"
        path.write_text(json.dumps(items))
        events = ms.json_events(path, ["TA", "TB"])
        csv = Path(folder) / "events.csv"
        csv.write_text("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,"
                       "Start X,Start Y,End X,End Y\n"
                       "Away,SET PIECE,KICK OFF,1,1,0.04,0,0,Player19,,NaN,NaN,NaN,NaN\n"
                       "Home,BALL LOST,HEAD-INTERCEPTION,1,5,0.2,9,0.36,Player1,,0.1,0.1,0.2,0.2\n"
                       "Home,PASS,,2,30,1.2,41,1.64,Player1,Player2,0.1,0.1,0.2,0.2\n")
        from_csv = ms.csv_events(csv)
    assert [(e.team, e.kind, e.subtypes, e.period, e.start, e.end) for e in events] == [
        (1, "SET PIECE", ("KICK OFF",), 1, 10, 10), (0, "BALL LOST", ("HEAD", "INTERCEPTION"), 2, 20, 20),
        (0, "PASS", (), 2, 30, 41)]
    assert [(e.team, e.kind, e.subtypes, e.period, e.start, e.end) for e in from_csv] == [
        (1, "SET PIECE", ("KICK OFF",), 1, 1, 0), (0, "BALL LOST", ("HEAD", "INTERCEPTION"), 1, 5, 9),
        (0, "PASS", (), 2, 30, 41)]


# -- targets ---------------------------------------------------------------------------------------

def test_marks_hand_possession_over_at_the_end_of_a_ball_lost():
    events = [event(0, 0, "PASS", 110, 120), event(1, 0, "BALL LOST", 130, 140), event(2, 1, "CHALLENGE", 132),
              event(3, 1, "RECOVERY", 140), event(4, 1, "CARD", 150), event(5, 0, "SHOT", 160)]
    assert ms.marks(events, 100, 100) == [(10, 0), (30, 0), (40, 1), (40, 1), (60, 0)]


def test_marks_without_the_handover_wait_for_the_opponents_next_event():
    events = [event(0, 0, "PASS", 110, 120), event(1, 0, "BALL LOST", 130, 140), event(2, 1, "PASS", 150, 155)]
    assert ms.marks(events, 100, 100) == [(10, 0), (30, 0), (40, 1), (50, 1)]
    assert ms.marks(events, 100, 100, handover=frozenset()) == [(10, 0), (30, 0), (50, 1)]
    visible = np.ones(100, dtype=bool)
    events = [event(0, 0, "SET PIECE", 100, subtypes=("KICK OFF",)), *events]
    d19, _ = ms.raw_labels(visible, events, 100)
    second, _ = ms.raw_labels(visible, events, 100, handover=frozenset())
    assert (d19[40:50] == A).all() and (second[40:50] == H).all() and (second[50:] == A).all()
    np.testing.assert_array_equal(d19 == D, second == D)


def test_raw_labels_from_ball_and_events():
    first = 1000
    visible = np.ones(400, dtype=bool)
    visible[:20] = False           # before the kick-off
    visible[100:110] = False       # a 10-frame dropout inside play: live
    visible[200:260] = False       # a stoppage
    visible[350:] = False          # after the last visible ball
    events = [event(0, 0, "SET PIECE", first + 20, subtypes=("KICK OFF",)), event(1, 0, "PASS", first + 20, first + 40),
              event(2, 0, "BALL LOST", first + 150, first + 170), event(3, 1, "BALL OUT", first + 190, first + 199),
              # the restart comes 5 frames after the ball reappears; those frames take the restart team
              event(4, 0, "SET PIECE", first + 265, subtypes=("THROW IN",)), event(5, 0, "PASS", first + 265, first + 280)]
    labels, counts = ms.raw_labels(visible, events, first)
    expected = np.full(400, X, dtype=np.int8)
    expected[20:170], expected[170:200], expected[200:260], expected[260:350] = H, A, D, H
    np.testing.assert_array_equal(labels, expected)
    assert counts["kick_off"] == 20 and counts["first_seen"] == 20 and counts["last_seen"] == 349
    assert counts["dropouts_filled"] == 1 and counts["dropout_frames"] == 10 and counts["segments_without_marks"] == 0


def test_long_missing_runs_stay_dead_and_short_ones_live():
    visible = np.ones(100, dtype=bool)
    visible[10:34] = False   # 24 frames: dropout
    visible[50:75] = False   # 25 frames: one second, dead
    live, filled = ms.live_mask(visible)
    assert live[10:34].all() and not live[50:75].any() and filled == [24]


def test_control_rule():
    first = 0
    raw = np.full(300, H, dtype=np.int8)
    raw[0], raw[-1] = X, X
    raw[50:55] = A     # A touches for 5 frames before losing it: no control, back to H
    raw[100:160] = A   # A holds for 60 frames before a pass: control
    raw[200:205] = A   # A's one-frame carry then loses it: no control
    raw[240:250] = A   # A's 0.6 s carry: control
    events = [event(0, 0, "SET PIECE", 1, subtypes=("KICK OFF",)),
              event(1, 1, "RECOVERY", 50), event(2, 1, "BALL LOST", 52, 55),
              event(3, 1, "RECOVERY", 100), event(4, 1, "PASS", 150, 155),
              event(5, 1, "RECOVERY", 200), event(6, 1, "CARRY", 200, 201), event(7, 1, "BALL LOST", 202, 205),
              event(8, 1, "RECOVERY", 240), event(9, 1, "CARRY", 240, 255), event(10, 1, "BALL LOST", 245, 250)]
    control = ms.control_labels(raw, events, first)
    expected = raw.copy()
    expected[50:55], expected[200:205] = H, H
    np.testing.assert_array_equal(control, expected)
    # A spell is controlled by a hold of at least 0.5 s even without a pass or carry.
    assert ms.controlled(1, 50, 64, [event(1, 1, "RECOVERY", 50), event(2, 1, "BALL LOST", 63, 64)], first)
    assert not ms.controlled(1, 50, 62, [event(1, 1, "RECOVERY", 50), event(2, 1, "BALL LOST", 62, 62)], first)
    # The first spell of a live segment is always controlled (a restart has a clear possessor).
    restart = np.r_[np.full(5, D), np.full(3, A), np.full(20, H)].astype(np.int8)
    np.testing.assert_array_equal(ms.control_labels(restart, [event(0, 0, "RECOVERY", 8)], 0), restart)


def test_reader_checks_count_missing_ball_in_flight_and_before_restarts():
    visible = np.ones(300, dtype=bool)
    visible[100:160] = False
    visible[30] = False
    events = [event(0, 0, "SET PIECE", 0, subtypes=("KICK OFF",)), event(1, 0, "PASS", 20, 39),
              event(2, 1, "SET PIECE", 160, subtypes=("THROW IN",)), event(3, 1, "SET PIECE", 250, subtypes=("FREE KICK",))]
    checks = ms.reader_checks(visible, events, 0)
    assert checks == {"flight_frames": 20, "flight_missing": 1, "restarts": 2, "restarts_after_missing": 1}
    verdict = ms.convention_passed([checks])
    assert verdict["passed"] is False and verdict["restarts_after_missing_share"] == .5


# -- scoring and veto ------------------------------------------------------------------------------

def synthetic_half(seed, length=600):
    rng = np.random.default_rng(seed)
    labels = np.full(length, H, dtype=np.int8)
    labels[:10], labels[-10:] = X, X
    labels[150:300], labels[300:350], labels[450:500] = A, D, A
    positions = rng.uniform(-30, 30, (length, 4, 2)).astype(np.float32)
    ball = rng.uniform(-30, 30, (length, 2)).astype(np.float32)
    ball[labels == D] = np.nan
    return {"positions": positions, "ball": ball, "teams": np.array([0, 0, 1, 1], dtype=np.int8),
            "pitch": np.array([105.0, 68.0]), "labels": labels, "control": labels.copy()}


def test_targets_and_scores_cover_every_convention():
    halves = [synthetic_half(1), synthetic_half(2)]
    truth = ms.targets(halves)
    assert tuple(truth) == ms.TARGETS and len(truth["raw"]) == 1201 and truth["raw"][600] == X
    perfect = ms.score(truth, [halves[0]["labels"], halves[1]["labels"]])
    assert all(perfect[name]["spell_f1"] == 1.0 for name in ("raw", "control"))
    np.testing.assert_array_equal(ms.swapped([H, A, D, X]), [A, H, D, X])


def test_sanity_needs_range_orientation_and_conventions():
    def scores(spell, macro):
        return {"raw": {"spell_f1": spell, "macro_f1": macro}, **{name: {"spell_f1": spell, "macro_f1": macro}
                                                                  for name in ms.TARGETS[1:]}}
    good = {"passed": True}
    baseline = {"1": scores(.3, .6), "2": scores(.32, .62)}
    swap = {"1": scores(.1, .3), "2": scores(.1, .3)}
    assert ms.sanity_passed(baseline, swap, {"1": good, "2": good})[0]
    assert not ms.sanity_passed(baseline, {"1": scores(.1, .7), "2": scores(.1, .3)}, {"1": good, "2": good})[0]
    assert not ms.sanity_passed({"1": scores(.6, .6), "2": scores(.6, .6)}, swap, {"1": good, "2": good})[0]
    assert not ms.sanity_passed(baseline, swap, {"1": good, "2": {"passed": False}})[0]


def test_veto_against_cached_d3():
    from dfl_submission import diagnosis
    previous = os.environ.get("MTAH_DATA_DIR")
    os.environ["MTAH_DATA_DIR"] = "/nonexistent/mtah-test"
    try:
        with TemporaryDirectory() as folder:
            parsed, cache, record = Path(folder) / "parsed", Path(folder) / "cache", Path(folder) / "d3.json"
            parsed.mkdir()
            entries, halves, d3 = {}, [], {}
            for game in ms.GAMES:
                d3[game] = []
                for index in (1, 2):
                    half = synthetic_half(10 * game + index)
                    path = parsed / f"{game}_{index}.npz"
                    np.savez_compressed(path, **{name: half[name] for name in ms.ARRAYS})
                    entries[path.name] = list(ms.identity(path))
                    prediction = half["labels"].copy()
                    prediction[prediction == X] = H
                    prediction[200:210] = H  # D3 is not perfect
                    d3[game].append(prediction)
                    signals = tuple(np.full(len(prediction), value) for value in (.5, .1, 0., 0.))
                    halves.append(diagnosis.HalfSignals("metrica", str(game), "observed", index - 1, half["labels"],
                                                        signals, prediction))
            (parsed / "manifest.json").write_text(json.dumps({"reader_sha256": "test", "parsed": entries}))
            identity_record = {"protocol": ms.PROTOCOL, "teacher_sha256": ms.teacher_identity(),
                               "inputs": ms.inputs(parsed)}
            diagnosis.collect(cache, identity_record, lambda: iter(halves))
            truths = {game: ms.targets(ms.load(parsed, game)) for game in ms.GAMES}
            matches = {str(game): ms.score(truths[game], d3[game]) for game in ms.GAMES}
            record.write_text(json.dumps({"matches": matches}))

            same = ms.veto({game: d3[game] for game in ms.GAMES}, reference="D3", parsed=parsed, cache=cache, record=record)
            assert same["changes"]["primary"] == 0 and same["changes"]["passed"]
            letters = {str(game): [np.array(pm.LABELS)[codes] for codes in d3[game]] for game in ms.GAMES}
            assert ms.veto(letters, reference="D3", parsed=parsed, cache=cache, record=record)["changes"]["primary"] == 0
            through_inputs = ms.veto(lambda half: half.d3, reference="D3", parsed=parsed, cache=cache, record=record)
            assert through_inputs["changes"]["targets"] == same["changes"]["targets"]
            worse = ms.veto({game: [np.full(600, D, dtype=np.int8)] * 2 for game in ms.GAMES},
                            reference="D3", parsed=parsed, cache=cache, record=record)
            assert worse["changes"]["primary"] < -ms.VETO_LOSS and not worse["changes"]["passed"]
            decoded = ms.veto({game: [half.signals for half in halves if half.match == str(game)] for game in ms.GAMES},
                              reference="D3", parsed=parsed, cache=cache, record=record)
            assert set(decoded["changes"]["targets"]) == set(ms.TARGETS)
            altered = json.loads(json.dumps(matches))
            altered["1"]["aba3"]["spell_f1"] += .001
            record.write_text(json.dumps({"matches": altered}))
            try:
                ms.veto({game: d3[game] for game in ms.GAMES}, reference="D3", parsed=parsed, cache=cache, record=record)
            except ValueError as error:
                assert "reproduce" in str(error)
            else:
                raise AssertionError("A D3 record that the cache does not reproduce must stop the veto")
    finally:
        if previous is None:
            os.environ.pop("MTAH_DATA_DIR")
        else:
            os.environ["MTAH_DATA_DIR"] = previous


def test_veto_defaults_to_the_standing_reference_and_its_recipe():
    from dfl_submission import diagnosis
    previous = os.environ.get("MTAH_DATA_DIR")
    original = ms.reference_identity, ms.reference_recipe
    recipes = {name: {"cutoff": .53, "dead_cost": 2, "owner_cost": cost, "discount": .8}
               for name, cost in (("D3", 6), ("D15", 8))}
    os.environ["MTAH_DATA_DIR"] = "/nonexistent/mtah-test"
    ms.reference_identity = lambda reference, parsed: {"reference": reference, "inputs": ms.inputs(parsed)}
    ms.reference_recipe = lambda reference: recipes[reference]
    try:
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            parsed = folder / "parsed"
            parsed.mkdir()
            entries, truths = {}, {}
            for game in ms.GAMES:
                for index in (1, 2):
                    half = synthetic_half(10 * game + index)
                    path = parsed / f"{game}_{index}.npz"
                    np.savez_compressed(path, **{name: half[name] for name in ms.ARRAYS})
                    entries[path.name] = list(ms.identity(path))
            (parsed / "manifest.json").write_text(json.dumps({"reader_sha256": "test", "parsed": entries}))
            truths = {game: ms.targets(ms.load(parsed, game)) for game in ms.GAMES}
            rng = np.random.default_rng(5)
            predicted, caches, records = {}, {}, {}
            for name, damage in (("D3", 200), ("D15", 100)):
                halves, predicted[name] = [], {}
                for game in ms.GAMES:
                    predicted[name][game] = []
                    for index, half in enumerate(ms.load(parsed, game)):
                        labels = np.where(half["labels"] == X, H, half["labels"]).astype(np.int8)
                        labels[damage:damage + 10] = A
                        signals = tuple(rng.uniform(0, 1, len(labels)) for _ in range(4))
                        halves.append(diagnosis.HalfSignals("metrica", str(game), "observed", index, half["labels"],
                                                            signals, labels))
                        predicted[name][game].append(labels)
                caches[name], records[name] = folder / name, folder / f"{name}.json"
                diagnosis.collect(caches[name], ms.reference_identity(name, parsed), lambda: iter(halves))
                records[name].write_text(json.dumps({"matches": {
                    str(game): ms.score(truths[game], predicted[name][game]) for game in ms.GAMES}}))
            assert ms.STANDING == "D15" and ms.cache_path("D15") == ms.CACHE / "d15"
            assert ms.record_path("D15").name == "d15_measurement.json"

            standing = ms.veto(predicted["D15"], parsed=parsed, cache=caches["D15"], record=records["D15"])
            assert standing["reference"]["name"] == "D15" and standing["changes"]["primary"] == 0
            older = ms.veto(predicted["D3"], parsed=parsed, cache=caches["D15"], record=records["D15"])
            d3_means = ms.means(json.loads(records["D3"].read_text())["matches"])
            expected = d3_means[ms.PRIMARY]["spell_f1"] - standing["reference"]["means"][ms.PRIMARY]["spell_f1"]
            assert abs(older["changes"]["primary"] - expected) < 1e-12
            # Decoder inputs are decoded with the reference's recipe unless one is given.
            inputs = {game: [half.signals for half in ms.reference_halves("D15", parsed, caches["D15"])[1]
                             if half.match == str(game)] for game in ms.GAMES}
            through_inputs = ms.veto(inputs, parsed=parsed, cache=caches["D15"], record=records["D15"])
            decoded = {game: [diagnosis.decode(part, recipes["D15"]) for part in inputs[game]] for game in ms.GAMES}
            assert through_inputs["candidate"] == ms.veto(decoded, parsed=parsed, cache=caches["D15"],
                                                          record=records["D15"])["candidate"]
            try:
                ms.veto(predicted["D15"], parsed=parsed, cache=caches["D15"], record=records["D3"])
            except ValueError as error:
                assert "D15" in str(error)
            else:
                raise AssertionError("A reference must reproduce its own record")
    finally:
        ms.reference_identity, ms.reference_recipe = original
        if previous is None:
            os.environ.pop("MTAH_DATA_DIR")
        else:
            os.environ["MTAH_DATA_DIR"] = previous

def test_pinned_sources_cover_every_file():
    pinned = json.loads(ms.SOURCES.read_text())
    assert pinned["commit"] == ms.COMMIT and set(pinned["files"]) == set(ms.pinned_paths())
    assert all(len(value) == 2 and len(value[1]) == 64 for value in pinned["files"].values())
