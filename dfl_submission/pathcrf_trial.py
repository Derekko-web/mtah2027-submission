"""D18: PathCRF ball-free possession check (DECISIONS.md D18).

`measure` runs one published PathCRF checkpoint (Kim et al., KDD 2026; trial 140) on the DFL WPY
guard views and the four SkillCorner development matches. Next to it are frozen D3's cached decoded
labels. It reports team-possession accuracy on target-live frames, all of them and those where the
view's ball is hidden, and the spellF1 of a hybrid that takes D3's stoppages and PathCRF's teams. It
selects nothing and changes nothing in the submission; it only decides whether B2 runs.

PathCRF's code (MPL-2.0) is not copied here. Its model classes are imported at run time from a
separate clone, whose files are checked against pinned SHA-256 hashes first. The checkpoint is read
with `weights_only=True`. Inference runs on CPU.
"""

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from dfl_submission.conventions import live_segments
from dfl_submission.external_only import require_external_only
from evaluation import possession_metrics as pm


PACKAGE = Path(__file__).resolve().parent
RECIPE = PACKAGE / "pathcrf_recipe"
PROTOCOL = "DECISIONS.md D18"
FPS = 25
STEP = 5                      # 25 Hz to PathCRF's 5 Hz
WINDOW = 50                   # 10 s at 5 Hz
STRIDE = 25                   # windows overlap by half
TEAM_SIZE = 11
OUTSIDE_NODES = 4
PITCH = (105.0, 68.0)         # PathCRF's frame, origin at a corner
OUTSIDE = ((0.0, PITCH[1] / 2), (PITCH[0], PITCH[1] / 2), (PITCH[0] / 2, 0.0), (PITCH[0] / 2, PITCH[1]))
SPEED_LIMIT = 12.0            # m/s, PathCRF's outlier rule
VELOCITY_FILTER = (15, 2)
ACCEL_FILTER = (9, 2)
MIN_FILTER_FRAMES = 17
GOALKEEPER_SHARE = .1
SANITY_MIN = .80
TARGETS = {"aba3": 3.0, "raw": None}
PRIMARY = "aba3"
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner")
SUBSETS = ("all", "ball_hidden", "ball_visible")
VISIBLE_BINS = {"under_11": (0, 11), "11_to_14": (11, 15), "15_plus": (15, 10 ** 6)}
COMMIT = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
TRIAL = 140
CHECKPOINT = "saved/140/model/state_dict_best_acc.pt"
PINNED = {
    "LICENSE": "3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04",
    CHECKPOINT: "55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9",
}
IMPORTED = ("saved/140/args.json", "datatools/config.py", "models/__init__.py", "models/set_lstm.py",
            "models/utils.py", "models/dynamic_dense_crf.py", "models/dynamic_sparse_crf.py",
            "models/static_dense_crf.py", "models/static_sparse_crf.py", "set_transformer/attention.py",
            "set_transformer/blocks.py", "set_transformer/model.py")
CLONE_MODULES = ("models", "datatools", "set_transformer")
H, A, D, X = pm.H, pm.A, pm.D, pm.X


# -- inputs ----------------------------------------------------------------------------------------

def interpolate_interior(positions):
    """Each slot's gaps between its first and last observation, filled linearly in time; nothing outside."""
    positions = np.array(positions, dtype=np.float64, copy=True)
    frames = np.arange(len(positions))
    for slot in range(positions.shape[1]):
        seen = np.isfinite(positions[:, slot]).all(axis=1)
        observed = np.flatnonzero(seen)
        inside = np.zeros(len(positions), dtype=bool)
        if len(observed):
            inside[observed[0]:observed[-1] + 1] = True
            for axis in range(2):
                positions[inside, slot, axis] = np.interp(frames[inside], observed, positions[observed, slot, axis])
        positions[~inside, slot] = np.nan
    return positions


def to_pathcrf(positions, pitch):
    """Centred metres to PathCRF's 105 x 68 m frame with the origin at a corner."""
    length, width = float(pitch[0]), float(pitch[1])
    result = np.empty_like(np.asarray(positions, dtype=np.float64))
    result[..., 0] = (positions[..., 0] / length + .5) * PITCH[0]
    result[..., 1] = (positions[..., 1] / width + .5) * PITCH[1]
    return result


def running_features(xy):
    """x, y, vx, vy, speed and acceleration magnitude of one continuous track, as PathCRF computes them."""
    from scipy.signal import savgol_filter
    xy = np.array(xy, dtype=np.float64, copy=True)
    count = len(xy)
    result = np.zeros((count, 6))
    if count < 2:
        result[:, :2] = xy
        return result
    jump = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T) * FPS]
    keep = jump <= SPEED_LIMIT
    if not keep.all():
        frames = np.arange(count)
        for axis in range(2):
            xy[:, axis] = np.interp(frames, frames[keep], xy[keep, axis])
    result[:, :2] = xy
    velocity = np.diff(xy, axis=0) * FPS
    if count >= MIN_FILTER_FRAMES:
        velocity = savgol_filter(velocity, *VELOCITY_FILTER, axis=0)
        accel = savgol_filter(np.diff(velocity, axis=0) * FPS, *ACCEL_FILTER, axis=0)
    else:
        accel = np.diff(velocity, axis=0) * FPS
    result[1:, 2:4] = velocity
    result[0, 2:4] = velocity[0]
    result[:, 4] = np.hypot(result[:, 2], result[:, 3])
    if count >= 3:
        result[1:-1, 5] = np.hypot(accel[:, 0], accel[:, 1])
    return result


def half_features(positions, pitch):
    """(frames, slots, 6) PathCRF features, NaN outside each slot's observed span."""
    filled = to_pathcrf(interpolate_interior(positions), pitch)
    features = np.full((*filled.shape[:2], 6), np.nan)
    for slot in range(filled.shape[1]):
        index = np.flatnonzero(np.isfinite(filled[:, slot]).all(axis=1))
        if len(index):
            features[index, slot] = running_features(filled[index, slot])
    return features


def goalkeepers(features, teams):
    """Per team, the slot whose mean x is farthest from the centre line, among slots seen on 10% of frames."""
    teams = np.asarray(teams)
    seen = np.isfinite(features[..., 0])
    result = {}
    for team in (0, 1):
        slots = [slot for slot in np.flatnonzero(teams == team) if seen[:, slot].sum() >= GOALKEEPER_SHARE * len(features)]
        if not slots:
            raise ValueError(f"No goalkeeper candidate for team {team}")
        depth = [abs(np.nanmean(features[:, slot, 0]) - PITCH[0] / 2) for slot in slots]
        result[team] = int(slots[int(np.argmax(depth))])
    return result


def nearest_index(keep):
    """For every position, the index of the nearest True in `keep`, the earlier one on ties."""
    index = np.arange(len(keep))
    before = np.maximum.accumulate(np.where(keep, index, -1))
    after = np.minimum.accumulate(np.where(keep, index, len(keep))[::-1])[::-1]
    use_before = (before >= 0) & ((after >= len(keep)) | (index - before <= after - index))
    return np.where(use_before, before, after)


def downsample(length):
    steps = np.arange(0, length, STEP)
    if steps[-1] != length - 1:
        steps = np.r_[steps, length - 1]
    return steps


def segment_input(features, teams, keepers, start, stop):
    """PathCRF's (steps, 26, 8) input for one segment, its 25 Hz frame of each step and each node's team.

    Node features: team indicator (-1 first team, +1 second), goalkeeper flag, x, y, vx, vy, speed, acceleration.
    Nodes: the first team's players (the one whose goalkeeper is further left) padded to 11, the second
    team's padded to 11, then the four outside nodes. Node team is 0 home, 1 away, -1 pad or outside.
    """
    teams = np.asarray(teams)
    part = features[start:stop]
    seen = np.isfinite(part[..., 0])
    chosen = {}
    for team in (0, 1):
        slots = np.flatnonzero((teams == team) & seen.any(axis=0))
        counts = seen[:, slots].sum(axis=0)
        chosen[team] = sorted(slots[np.argsort(-counts, kind="stable")][:TEAM_SIZE].tolist())
    filled = part.copy()
    for slots in chosen.values():
        for slot in slots:
            known = seen[:, slot]
            if known.all():
                continue
            filled[:, slot, :2] = part[nearest_index(known), slot, :2]
            filled[~known, slot, 2:] = 0.0

    def mean_x(team):
        if keepers[team] in chosen[team]:
            return float(filled[:, keepers[team], 0].mean())
        return float(filled[:, chosen[team], 0].mean()) if chosen[team] else PITCH[0] / 2

    first = 0 if mean_x(0) <= mean_x(1) else 1
    steps = downsample(stop - start)
    nodes = np.zeros((len(steps), 2 * TEAM_SIZE + OUTSIDE_NODES, 8), dtype=np.float32)
    node_team = np.full(2 * TEAM_SIZE + OUTSIDE_NODES, -1, dtype=np.int64)
    for block, team in enumerate((first, 1 - first)):
        for k, slot in enumerate(chosen[team]):
            node = block * TEAM_SIZE + k
            nodes[:, node, 0] = -1.0 if block == 0 else 1.0
            nodes[:, node, 1] = 1.0 if slot == keepers[team] else 0.0
            nodes[:, node, 2:] = filled[steps, slot]
            node_team[node] = team
    for k, (x, y) in enumerate(OUTSIDE):
        nodes[:, 2 * TEAM_SIZE + k, 2] = x
        nodes[:, 2 * TEAM_SIZE + k, 3] = y
    return nodes, steps, node_team


# -- inference -------------------------------------------------------------------------------------

def windows(length):
    """10 s windows overlapping by half; the last one ends at the segment's end."""
    if length <= WINDOW:
        return [(0, length)]
    starts = list(range(0, length - WINDOW + 1, STRIDE))
    if starts[-1] != length - WINDOW:
        starts.append(length - WINDOW)
    return [(start, start + WINDOW) for start in starts]


def window_weights(spans, i):
    """Linear ramps across each overlap, so neighbouring windows blend; never exactly zero."""
    start, stop = spans[i]
    weight = np.ones(stop - start)
    if i > 0:
        overlap = min(spans[i - 1][1] - start, stop - start)
        if overlap > 0:
            weight[:overlap] = np.minimum(weight[:overlap], np.linspace(0, 1, overlap))
    if i < len(spans) - 1:
        overlap = min(stop - spans[i + 1][0], stop - start)
        if overlap > 0:
            weight[-overlap:] = np.minimum(weight[-overlap:], np.linspace(1, 0, overlap))
    return np.maximum(weight, 1e-3)


def segment_edges(model, nodes):
    """Decoded (sender, receiver) node of every step: blended window outputs, then the CRF's Viterbi path."""
    import torch
    valid = model.valid_orig_edge_ids
    spans = windows(len(nodes))
    emissions = embeds = None
    total = torch.zeros(len(nodes))
    with torch.no_grad():
        for i, (start, stop) in enumerate(spans):
            _, logits, edge_h = model(torch.as_tensor(nodes[start:stop]).unsqueeze(0))
            weight = torch.as_tensor(window_weights(spans, i), dtype=torch.float32)
            logits = logits[0][:, valid]
            edge_h = edge_h[0][:, valid]
            if emissions is None:
                emissions = torch.zeros(len(nodes), logits.shape[-1])
                embeds = torch.zeros(len(nodes), *edge_h.shape[1:])
            emissions[start:stop] += logits * weight[:, None]
            embeds[start:stop] += edge_h * weight[:, None, None]
            total[start:stop] += weight
        emissions /= total[:, None]
        embeds /= total[:, None, None]
        tags = model.crf.decode(emissions.unsqueeze(0), embeds.unsqueeze(0)).squeeze(0)
    return model.comp_src[tags].numpy(), model.comp_dst[tags].numpy()


def step_teams(sender, receiver, node_team):
    """Team of every step: the holder's or passer's; across teams, passer then receiver; none when out of play."""
    sender, receiver = np.asarray(sender), np.asarray(receiver)
    players = 2 * TEAM_SIZE
    from_team = node_team[sender]
    to_team = np.where(receiver < players, node_team[receiver], -1)
    result = np.where(sender < players, from_team, -1)
    across = (sender < players) & (receiver < players) & (from_team >= 0) & (to_team >= 0) & (from_team != to_team)
    edge = sender * 1000 + receiver
    start = 0
    for i in range(1, len(edge) + 1):
        if i == len(edge) or edge[i] != edge[start]:
            if across[start]:
                run = i - start
                late = np.arange(run) >= run / 2
                result[start:i] = np.where(late, to_team[start], from_team[start])
            start = i
    return result


def frame_teams(teams, steps, length):
    """Hold each step's team until the next step."""
    return np.asarray(teams)[np.searchsorted(steps, np.arange(length), side="right") - 1]


def pathcrf_labels(model, features, teams, keepers, segments, length):
    """H/A on frames inside the segments where PathCRF gives a team, D elsewhere."""
    result = np.full(length, D, dtype=np.int8)
    for start, stop in segments:
        nodes, steps, node_team = segment_input(features, teams, keepers, start, stop)
        sender, receiver = segment_edges(model, nodes)
        team = frame_teams(step_teams(sender, receiver, node_team), steps, stop - start)
        result[start:stop] = np.where(team == 0, H, np.where(team == 1, A, D))
    return result


# -- measures --------------------------------------------------------------------------------------

def team_home(labels):
    return np.where(labels == H, 1.0, np.where(labels == A, 0.0, .5))


def hybrid(d3, pathcrf):
    """D3's labels with the team replaced by PathCRF's wherever both give one."""
    both = np.isin(d3, (H, A)) & np.isin(pathcrf, (H, A))
    return np.where(both, pathcrf, d3).astype(np.int8)


def team_counts(truth, prediction, home, mask):
    """(correct, frames) of the team on target-live frames under `mask`, with D11's true-stoppage fill."""
    from dfl_submission.diagnosis import oracle_stoppage
    filled = oracle_stoppage(truth, prediction, home)
    live = np.isin(truth, (H, A)) & mask
    return int((filled[live] == truth[live]).sum()), int(live.sum())


def ratio(counts):
    correct, total = counts
    return correct / total if total else None


def add(first, second):
    return first[0] + second[0], first[1] + second[1]


def visible_players(positions):
    return np.isfinite(np.asarray(positions)).all(axis=2).sum(axis=1)


# -- the D3 cache ----------------------------------------------------------------------------------

def cached_halves(cache):
    """D3's cached DFL and SkillCorner development halves. The PFF entries are never opened."""
    from dfl_submission.diagnosis import load_half
    entries = json.loads((cache / "manifest.json").read_text())["halves"]
    return {(entry["source"], entry["match"], entry["view"], entry["index"]): load_half(cache, entry)
            for entry in entries if entry["source"] in ("dfl", "skillcorner")}


def group_of(source, view):
    return source if source != "dfl" else f"dfl_{view}"


def match_arrays(halves, keys, predict):
    """Truth, prediction and home over both halves of a match, with an X frame between them."""
    first, second = (halves[key] for key in keys)
    truth = np.r_[first.labels, X, second.labels].astype(np.int8)
    prediction = np.r_[predict(keys[0]), X, predict(keys[1])].astype(np.int8)
    home = np.r_[first.signals[0], np.nan, second.signals[0]]
    return truth, prediction, home


def matches_of(halves):
    grouped = {}
    for source, match, view, index in sorted(halves):
        grouped.setdefault((source, match, view), []).append((source, match, view, index))
    if any([key[3] for key in keys] != [0, 1] for keys in grouped.values()):
        raise ValueError("Expected two halves per match")
    return grouped


def reproduction(halves, d11):
    """D3 decoded from the cache against D11's group spellF1 and true-stoppage oracle means."""
    from dfl_submission.diagnosis import oracle_stoppage, score, target
    grouped = matches_of(halves)
    result = {"passed": True, "groups": {}}
    for name, seconds in TARGETS.items():
        for group in GROUPS:
            spells, oracle = [], []
            for match_key in sorted(key for key in grouped if group_of(key[0], key[2]) == group):
                truth, prediction, home = match_arrays(halves, grouped[match_key], lambda key: halves[key].d3)
                truth = target(truth, seconds)
                spells.append(score(truth, prediction)["spell_f1"])
                oracle.append(score(truth, oracle_stoppage(truth, prediction, home))["spell_f1"])
            expected = d11["groups"][name][group]
            found = {"spell_f1_mean": float(np.mean(spells)), "oracle_stoppage": float(np.mean(oracle))}
            same = (found["spell_f1_mean"] == expected["spell_f1_mean"]
                    and found["oracle_stoppage"] == expected["oracle"]["stoppage"])
            result["groups"][f"{name}/{group}"] = found | {"reproduced": same}
            result["passed"] &= same
    return result


# -- PathCRF ---------------------------------------------------------------------------------------

def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def clone_identity(clone):
    """The clone's commit and the hashes of every file D18 reads or imports, checked against the pins."""
    head = (clone / ".git" / "HEAD").read_text().strip()
    commit = (clone / ".git" / head.split(" ", 1)[1]).read_text().strip() if head.startswith("ref:") else head
    if commit != COMMIT:
        raise ValueError(f"PathCRF clone is at {commit}, not the pinned {COMMIT}")
    hashes = {name: digest(clone / name) for name in (*PINNED, *IMPORTED)}
    for name, expected in PINNED.items():
        if hashes[name] != expected:
            raise ValueError(f"PathCRF file differs from its pin: {name}")
    return {"commit": commit, "trial": TRIAL, "checkpoint": CHECKPOINT, "sha256": hashes}


def load_model(clone, identity, expected=None):
    """Trial 140's SetLSTM with its dynamic masked CRF, built from the clone's classes, on CPU, in eval mode."""
    import torch
    if expected is not None and identity["sha256"] != expected:
        raise ValueError("PathCRF files differ from the recorded identity")
    clashes = [name for name in CLONE_MODULES if name in sys.modules]
    if clashes:
        raise ValueError(f"Modules already imported under PathCRF's names: {clashes}")
    if "torch_geometric" not in sys.modules:
        try:
            import torch_geometric  # noqa: F401  (only the GAT model, which D18 does not use, needs it)
        except ImportError:
            stub, data = types.ModuleType("torch_geometric"), types.ModuleType("torch_geometric.data")
            data.Batch = data.Data = object
            stub.data = data
            sys.modules["torch_geometric"], sys.modules["torch_geometric.data"] = stub, data
    sys.path.insert(0, str(clone))
    try:
        from models.set_lstm import SetLSTM
    finally:
        sys.path.remove(str(clone))
    for name in CLONE_MODULES:
        module = sys.modules.get(name)
        location = getattr(module, "__file__", None) or next(iter(getattr(module, "__path__", [])), "")
        if module is not None and not Path(location).resolve().is_relative_to(clone.resolve()):
            raise ValueError(f"{name} was not imported from the PathCRF clone")
    args = json.loads((clone / "saved/140/args.json").read_text())
    model = SetLSTM(macro_type=args["macro_type"], micro_type=args["micro_type"], seq_model_type=args["seq_model"],
                    crf_model_type=args["crf_model"], feat_dim=args["node_in_dim"], team_size=args["team_size"],
                    macro_ppe_dim=args["coarse_dim"], macro_fpe_dim=args["coarse_dim"], micro_enc_dim=args["fine_dim"],
                    macro_seq_dim=args["seq_dim"], micro_seq_dim=args["seq_dim"], crf_edge_dim=args["crf_edge_dim"],
                    dropout=args["dropout"])
    model.load_state_dict(torch.load(clone / CHECKPOINT, map_location="cpu", weights_only=True))
    if model.team_size != TEAM_SIZE or model.feat_dim != 8:
        raise ValueError("Unexpected PathCRF configuration")
    return model.eval()


# -- the measurement -------------------------------------------------------------------------------

def view_halves(dfl, skill):
    """The guard views of WPY and the SkillCorner development halves, keyed like the cache."""
    from dfl_submission import augment
    return {(half.source, half.match, half.view, half.index): half
            for half in augment.guard_halves(dfl, skill, "development", raw=True)}


def native_source(key):
    """The dead-ball-carried view moves only the ball, so PathCRF's input there is the masked view's."""
    source, match, view, index = key
    return (source, match, "masked", index) if view == "dead_carried" else key


def sanity(model, views, halves):
    """PathCRF's native team accuracy on WPY clean against the N = 3 targets (the adaptor gate)."""
    from dfl_submission.diagnosis import target
    counts, labels = (0, 0), {}
    for key in sorted(key for key in views if key[0] == "dfl" and key[2] == "clean"):
        half = views[key]
        truth = target(np.asarray(halves[key].labels, dtype=np.int8), TARGETS[PRIMARY])
        features = half_features(half.positions, half.pitch)
        prediction = pathcrf_labels(model, features, half.teams, goalkeepers(features, half.teams),
                                    list(live_segments(truth)), len(truth))
        labels[key] = prediction
        counts = add(counts, team_counts(truth, prediction, team_home(prediction), np.ones(len(truth), bool)))
    return {"accuracy": ratio(counts), "frames": counts[1], "minimum": SANITY_MIN,
            "passed": ratio(counts) >= SANITY_MIN}, labels


def run_all(model, views, halves, native=None):
    """Native and realistic PathCRF labels for every half."""
    from dfl_submission.diagnosis import target
    native = dict(native or {})
    realistic = {}
    features_of = {}
    for key in sorted(halves):
        half = views[key]
        source_key = native_source(key)
        if source_key != key and not np.array_equal(views[source_key].positions, half.positions, equal_nan=True):
            raise ValueError(f"{key} positions differ from {source_key}")
        if source_key not in features_of:
            features = half_features(views[source_key].positions, half.pitch)
            features_of[source_key] = (features, goalkeepers(features, half.teams))
        features, keepers = features_of[source_key]
        truth = target(np.asarray(halves[key].labels, dtype=np.int8), TARGETS[PRIMARY])
        if source_key not in native:
            native[source_key] = pathcrf_labels(model, features, half.teams, keepers, list(live_segments(truth)),
                                                len(truth))
        realistic[key] = pathcrf_labels(model, features, half.teams, keepers, list(live_segments(halves[key].d3)),
                                        len(truth))
        print(f"PathCRF {'/'.join(map(str, key))}", flush=True)
    return {key: native[native_source(key)] for key in halves}, realistic


def accuracy_record(views, halves, native, realistic):
    """Team-possession accuracy per group, match and subset, for D3, native PathCRF and the hybrid."""
    from dfl_submission.diagnosis import target
    models = ("d3", "pathcrf_native", "hybrid")
    pooled, per_match, by_visible = {}, {}, {}
    for key in sorted(halves):
        half, cached = views[key], halves[key]
        group = group_of(key[0], key[2])
        hidden = ~np.isfinite(np.asarray(half.ball)).all(axis=1)
        masks = {"all": np.ones(len(hidden), bool), "ball_hidden": hidden, "ball_visible": ~hidden}
        mixed = hybrid(cached.d3, realistic[key])
        predictions = {"d3": (cached.d3, cached.signals[0]), "pathcrf_native": (native[key], team_home(native[key])),
                       "hybrid": (mixed, cached.signals[0])}
        visible = visible_players(half.positions)
        for name, seconds in TARGETS.items():
            truth = target(np.asarray(cached.labels, dtype=np.int8), seconds)
            for subset, mask in masks.items():
                for model_name in models:
                    counts = team_counts(truth, *predictions[model_name], mask)
                    for table, label in ((pooled, group), (per_match, "/".join(map(str, key[:3])))):
                        slot = table.setdefault(name, {}).setdefault(label, {}).setdefault(subset, {})
                        slot[model_name] = add(slot.get(model_name, (0, 0)), counts)
            if key[0] == "skillcorner":
                for label, (low, high) in VISIBLE_BINS.items():
                    mask = hidden & (visible >= low) & (visible < high)
                    for model_name in models:
                        counts = team_counts(truth, *predictions[model_name], mask)
                        slot = by_visible.setdefault(name, {}).setdefault(label, {})
                        slot[model_name] = add(slot.get(model_name, (0, 0)), counts)

    def finish(table):
        return {name: {label: {subset: {"frames": next(iter(values.values()))[1],
                                        **{model: ratio(counts) for model, counts in values.items()}}
                               for subset, values in subsets.items()}
                       for label, subsets in labels.items()}
                for name, labels in table.items()}

    visible_table = {name: {label: {"frames": next(iter(values.values()))[1],
                                    **{model: ratio(counts) for model, counts in values.items()}}
                            for label, values in labels.items()}
                     for name, labels in by_visible.items()}
    return {"groups": finish(pooled), "matches": finish(per_match), "skillcorner_ball_hidden_by_visible": visible_table}


def spell_record(halves, realistic):
    """spellF1 and macroF1 of D3 and the hybrid per match and group, and the standard combined aggregate."""
    from dfl_submission.diagnosis import combined, score, target
    grouped = matches_of(halves)
    record = {}
    for name, seconds in TARGETS.items():
        per_match = {}
        for match_key, keys in sorted(grouped.items()):
            truth, d3, _ = match_arrays(halves, keys, lambda key: halves[key].d3)
            _, mixed, _ = match_arrays(halves, keys, lambda key: hybrid(halves[key].d3, realistic[key]))
            truth = target(truth, seconds)
            per_match[match_key] = {"d3": score(truth, d3), "hybrid": score(truth, mixed)}
        groups = {}
        for group in GROUPS:
            members = [value for key, value in per_match.items() if group_of(key[0], key[2]) == group]
            groups[group] = {model: {metric: float(np.mean([value[model][metric] for value in members]))
                                     for metric in ("spell_f1", "macro_f1")} for model in ("d3", "hybrid")}
        standard = {model: {metric: combined({key: {metric: value[model][metric]} for key, value in per_match.items()},
                                             metric)["combined"]
                            for metric in ("spell_f1", "macro_f1")}
                    for model in ("d3", "hybrid")}
        record[name] = {"groups": groups, "standard_combined": standard,
                        "matches": {"/".join(key): value for key, value in per_match.items()}}
    return record


def trigger(accuracy):
    """B2 runs only if native PathCRF beats D3 on N = 3 ball-hidden accuracy on WPY masked and on SkillCorner."""
    groups = accuracy["groups"][PRIMARY]
    tests = {}
    for name, group in (("T1_wpy_masked", "dfl_masked"), ("T2_skillcorner", "skillcorner")):
        values = groups[group]["ball_hidden"]
        tests[name] = {"frames": values["frames"], "d3": values["d3"], "pathcrf_native": values["pathcrf_native"],
                       "difference": values["pathcrf_native"] - values["d3"],
                       "passed": values["pathcrf_native"] > values["d3"]}
    return {"tests": tests, "b2_triggered": all(test["passed"] for test in tests.values())}


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def measure(clone, dfl, skill, cache, *, output=RECIPE / "d18_measurement.json"):
    require_external_only()
    import torch
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError(f"{output} is written once")
    d11 = json.loads((PACKAGE / "diagnosis_recipe" / "d11.json").read_text())
    halves = cached_halves(cache)
    check = reproduction(halves, d11)
    if not check["passed"]:
        raise SystemExit(f"D3 decoded from the cache does not reproduce D11: {check}")
    print("D3 reproduces the D11 record", flush=True)
    views = view_halves(dfl, skill)
    for key, cached in halves.items():
        if not np.array_equal(np.asarray(views[key].labels, dtype=np.int8), cached.labels):
            raise SystemExit(f"Labels of {key} differ from the cache")
    identity = clone_identity(clone)
    model = load_model(clone, identity)
    gate, clean_native = sanity(model, views, halves)
    print(f"Adaptor sanity gate: {gate}", flush=True)
    if not gate["passed"]:
        raise SystemExit("The adaptor sanity gate failed; no record is written (DECISIONS.md D18)")
    native, realistic = run_all(model, views, halves, clean_native)
    accuracy = accuracy_record(views, halves, native, realistic)
    record = {"protocol": PROTOCOL, "pathcrf": identity, "d3_cache": str(cache.name),
              "reproduction": check, "sanity": gate, "targets": {name: seconds for name, seconds in TARGETS.items()},
              "primary": PRIMARY, "accuracy": accuracy, "spells": spell_record(halves, realistic),
              "trigger": trigger(accuracy),
              "contamination": {"pathcrf_train": ["WMX", "WN1", "WOH", "WOY", "WPY"], "pathcrf_validation": ["WQQ"],
                                "pathcrf_test": ["WR9"], "scored_here": ["WPY", "SkillCorner development"]}}
    write_new(output, record)
    print(json.dumps(record["trigger"], indent=2), flush=True)
    return record


# -- the post-hoc diagnostic (D18 amendment) -------------------------------------------------------

NOISE_SEED = 270000          # the masked view's generator: 270000 + 100 * match index + half index
NOISE = .2


def noise_only(clean, masked, seed):
    """The clean tracks plus exactly the masked view's per-frame noise (the generator's first draw)."""
    noisy = np.asarray(clean, dtype=np.float32) + np.random.default_rng(seed).normal(0, NOISE, np.shape(clean)).astype(np.float32)
    seen = np.isfinite(masked)
    if not np.array_equal(noisy[seen], np.asarray(masked)[seen]):
        raise ValueError("The masked view's noise is not reproduced")
    return noisy


def dropouts_only(clean, masked):
    """The clean tracks with exactly the masked view's missing blocks, and no noise."""
    result = np.array(clean, dtype=np.float32, copy=True)
    result[~np.isfinite(masked)] = np.nan
    return result


def diagnose(clone, dfl, cache, *, output=RECIPE / "d18_diagnostic.json"):
    """PathCRF's native accuracy on WPY with the masked view's noise alone and its dropouts alone."""
    require_external_only()
    import torch
    from dfl_submission.diagnosis import target
    from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError(f"{output} is written once")
    measurement = json.loads((RECIPE / "d18_measurement.json").read_text())
    halves = {key: half for key, half in cached_halves(cache).items() if key[0] == "dfl"}
    views = view_halves(dfl, None)
    for key, cached in halves.items():
        if not np.array_equal(np.asarray(views[key].labels, dtype=np.int8), cached.labels):
            raise SystemExit(f"Labels of {key} differ from the cache")
    identity = clone_identity(clone)
    if identity != measurement["pathcrf"]:
        raise SystemExit("PathCRF clone differs from the D18 measurement")
    model = load_model(clone, identity)
    match_index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(DEVELOPMENT)
    counts = {}
    for index in (0, 1):
        clean, masked = views[("dfl", DEVELOPMENT, "clean", index)], views[("dfl", DEVELOPMENT, "masked", index)]
        variants = {"noise_only": noise_only(clean.positions, masked.positions, NOISE_SEED + 100 * match_index + index),
                    "dropouts_only": dropouts_only(clean.positions, masked.positions)}
        hidden = ~np.isfinite(np.asarray(masked.ball)).all(axis=1)
        primary = target(np.asarray(halves[("dfl", DEVELOPMENT, "clean", index)].labels, dtype=np.int8),
                         TARGETS[PRIMARY])
        for variant, positions in variants.items():
            features = half_features(positions, clean.pitch)
            prediction = pathcrf_labels(model, features, clean.teams, goalkeepers(features, clean.teams),
                                        list(live_segments(primary)), len(primary))
            for name, seconds in TARGETS.items():
                truth = target(np.asarray(halves[("dfl", DEVELOPMENT, "clean", index)].labels, dtype=np.int8), seconds)
                for subset, mask in (("all", np.ones(len(truth), bool)), ("ball_hidden", hidden)):
                    slot = counts.setdefault(name, {}).setdefault(variant, {})
                    slot[subset] = add(slot.get(subset, (0, 0)), team_counts(truth, prediction, team_home(prediction),
                                                                             mask))
            print(f"PathCRF {variant} {index}", flush=True)
    groups = measurement["accuracy"]["groups"]
    record = {"protocol": f"{PROTOCOL} (post-hoc diagnostic)", "pathcrf": identity,
              "measurement_sha256": digest(RECIPE / "d18_measurement.json"),
              "accuracy": {name: {variant: {subset: {"frames": value[1], "pathcrf_native": ratio(value)}
                                            for subset, value in subsets.items()}
                                  for variant, subsets in variants_counts.items()}
                           for name, variants_counts in counts.items()},
              "reference": {name: {view: {subset: groups[name][f"dfl_{view}"][subset]
                                          for subset in ("all", "ball_hidden")}
                                   for view in ("clean", "masked")}
                            for name in TARGETS}}
    write_new(output, record)
    print(json.dumps(record["accuracy"], indent=2), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure", "diagnose"))
    parser.add_argument("--clone", type=Path, required=True, help="PathCRF clone at the pinned commit")
    parser.add_argument("--dfl", type=Path, required=True, help="folder of the pinned DFL XML files")
    parser.add_argument("--skill", type=Path, help="SkillCorner folder (measure only)")
    parser.add_argument("--cache", type=Path, required=True, help="D11 decoder-input cache of the development halves")
    args = parser.parse_args()
    if args.command == "measure":
        if args.skill is None:
            parser.error("measure needs --skill")
        measure(args.clone.resolve(), args.dfl, args.skill, args.cache)
    else:
        diagnose(args.clone.resolve(), args.dfl, args.cache)


if __name__ == "__main__":
    main()
