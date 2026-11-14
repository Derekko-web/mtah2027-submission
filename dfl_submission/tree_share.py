"""D59: the adopted stack's decoder inputs kept as members, re-mixed with a tree share that depends on the ball (DECISIONS.md D59).

Where the trees enter. Per half the stack forecasts, for each of its two model sets (roles), D3's trees (home and
stoppage probabilities from the long-context tables at every fifth frame, interpolated to every frame), the
temporal network slot at phase 0 (`temporal_models`, three seeds averaged) and the seed-phase network (`seed_models`,
three seeds averaged per phase, the phases padded and averaged). The two roles are averaged leaf by leaf
(`retrain_trial.mix`, weight 0.5), and then `motion_trial.blend_signals` mixes:
- home: trees with the temporal slot at D3's `owner_mix` 0.5 (the trees' share in this blend is 0.5), and that
  with the seed-phase home at D15's `owner_mix` 0.5, so the trees carry a quarter of the final home probability;
- stoppage: the same with the two `dead_mix` 0.5; the two edges: the temporal slot with the seed phase.
All of this is on the half's 25 Hz frames: every array has one value per frame of the tracking the forecast was
given.

H(w). On frames where that tracking has no ball position (x or y NaN at the frame; the view's own ball, after any
hiding, as the forecast received it), the trees' share in the first home blend is w instead of 0.5, the rest going
to the temporal slot: home = w * trees + (1 - w) * slot there. The second blend, the stoppage mix and the edges are
unchanged. w = None leaves every frame at 0.5 and is exactly `motion_trial.blend_signals`.

R. The members are kept for D3's trees and for D59's refitted trees side by side, so each candidate only re-mixes
and re-decodes.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import context_trial as ct, stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d59"
MEMBERS = SCRATCH / "members"
PROTOCOL = "DECISIONS.md D59"
PARTS = ("development", "metrica", "d24", "guards", "situation")
STACK_CACHE = st.T_SCRATCH / "cache" / "stack"        # the adopted stack's decoder inputs from D48's development
COSTS = (12, 14, 16)
REFERENCE_COST = 14
SHARES = (.25, 0.0)
CANDIDATES = {"H(0.25)": ("d3", .25), "H(0)": ("d3", 0.0), "R": ("r", None), "RH(0.25)": ("r", .25), "RH(0)": ("r", 0.0)}
REFERENCE = ("d3", None)
SELECTION_GAIN, PFF_GAIN, LOSS = .008, .010, .01
FIELDS = ("trees_d3", "trees_r", "late", "seed")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def seen(ball):
    return np.isfinite(np.asarray(ball, dtype=np.float64)).all(axis=1)


# -- the blend with H -----------------------------------------------------------------------------------

def blend_h(mixed, previous, recipe, visible=None, share=None):
    """`motion_trial.blend_signals` with the trees' share of the first home blend `share` on frames not `visible`."""
    baseline, candidates = mixed
    trees, neural = baseline
    old_home, old_dead, dead_edge, owner_edge = neural[previous["variant"]]
    home_seen = ct.blend(trees[0], old_home, previous["owner_mix"])
    if share is None:
        old_home = home_seen
    else:
        home_unseen = ct.blend(trees[0], old_home, 1 - share)
        old_home = np.where(np.asarray(visible, dtype=bool), home_seen, home_unseen)
    old_dead = ct.blend(trees[1], old_dead, previous["dead_mix"])
    if recipe["variant"] is not None:
        home, dead, de, oe = candidates[recipe["variant"]]
        old_home = ct.blend(old_home, home, recipe["owner_mix"])
        old_dead = ct.blend(old_dead, dead, recipe["dead_mix"])
        dead_edge = ct.blend(dead_edge, de, recipe["dead_mix"])
        owner_edge = ct.blend(owner_edge, oe, recipe["owner_mix"])
    return old_home, old_dead, dead_edge, owner_edge


def structure(members, tree_set):
    """The role-mixed forecast structure `blend_signals` takes, with D3's (`d3`) or the refitted (`r`) trees."""
    from dfl_submission import ensemble_trial
    trees = members["trees_d3"] if tree_set == "d3" else members["trees_r"]
    if trees is None:
        raise ValueError("These members hold no refitted trees")
    return (tuple(trees), {ensemble_trial.LATE: tuple(members["late"])}), {"seed_phase": tuple(members["seed"])}


def signals(members, tree_set, share, previous, recipe):
    return tuple(blend_h(structure(members, tree_set), previous, recipe, members["seen"], share))


# -- the members ------------------------------------------------------------------------------------------

class MemberForecaster(st.Forecaster):
    """The stack's forecast up to the blend: per half, the role-mixed trees (D3's and R's), slot and seed phase."""

    def __init__(self, models, table, *, refit=None):
        from dfl_submission import long_context
        super().__init__(models, table, with_d43=False)
        if len(self.names) != 1:
            raise ValueError("D59 keeps the members of one model")
        self.refit = None if refit is None else {role: long_context.load(refit[key])[0] for role, key in self.roles.items()}

    def members(self, positions, ball, teams, pitch):
        from dfl_submission import ensemble_trial, long_context, player_set as ps, retrain_trial, temporal_trial
        sign = self.sign_of(positions, ball, teams, pitch)
        phases, baselines, refits = [], {}, {}
        for offset in ensemble_trial.PHASES:
            shifted_positions, shifted_ball = positions[offset:], ball[offset:]
            frames, owner, dead = long_context.features(shifted_positions, shifted_ball, teams, pitch)
            length = len(ball) - offset
            player_tokens = ps.tokens(shifted_positions, shifted_ball, teams, pitch, frames)
            if offset == 0:
                for role in self.roles:
                    baselines[role], _ = temporal_trial.forecast_features(self.trees[role], {}, frames, owner, dead, length)
                    if self.refit is not None:
                        refits[role], _ = temporal_trial.forecast_features(self.refit[role], {}, frames, owner, dead, length)
            phases.append((offset, frames, owner, player_tokens, length))
        outputs = {}

        def run(net, index):
            key = (net.key, index)
            if key not in outputs:
                _, frames, owner, player_tokens, length = phases[index]
                outputs[key] = net.predict(frames, owner, player_tokens, length, sign)
            return outputs[key]

        def slot(nets, index):
            return run(nets[0], index) if len(nets) == 1 else retrain_trial.mean([run(net, index) for net in nets])

        roles = self.models[self.names[0]]
        per_role = []
        for role in ("current", "candidate"):
            original, late = roles[role]["temporal_models"], roles[role]["seed_models"]
            start_reference = (baselines[role], {ensemble_trial.LATE: slot(original, 0)})
            phase_seed = [tuple(np.pad(p, (offset, 0), mode="edge") for p in slot(late, index))
                          for index, (offset, *_) in enumerate(phases)]
            per_role.append((start_reference, {"seed_phase": ensemble_trial.mean(phase_seed)}))
        mixed = retrain_trial.mix(*per_role, self.frozen["weight"])
        trees_r = None if self.refit is None else retrain_trial.mix(refits["current"], refits["candidate"], self.frozen["weight"])
        return {"seen": seen(ball), "trees_d3": tuple(mixed[0][0]), "trees_r": None if trees_r is None else tuple(trees_r),
                "late": tuple(mixed[0][1][ensemble_trial.LATE]), "seed": tuple(mixed[1]["seed_phase"])}

    def member_fn(self):
        return lambda _, positions, ball, teams, pitch: (self.members(positions, ball, teams, pitch), None)


@dataclass
class MemberHalf:
    source: str
    match: str
    view: str
    index: int
    labels: np.ndarray
    members: dict

    @property
    def key(self):
        return self.source, self.match, self.view


def save_member(folder, half):
    from dfl_submission import diagnosis
    arrays = {"labels": np.asarray(half.labels), "seen": half.members["seen"]}
    for field in FIELDS:
        value = half.members[field]
        if value is not None:
            arrays |= {f"{field}_{i}": array for i, array in enumerate(value)}
    path = folder / diagnosis.cache_name(half.source, half.match, half.view, half.index)
    np.savez_compressed(path, **arrays)
    return {"source": half.source, "match": half.match, "view": half.view, "index": half.index, "file": path.name,
            "sha256": digest(path)}


def load_member(folder, entry):
    path = folder / entry["file"]
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Cached members differ from the manifest: {path.name}")
    with np.load(path, allow_pickle=False) as item:
        members = {"seen": item["seen"]}
        for field, count in (("trees_d3", 2), ("trees_r", 2), ("late", 4), ("seed", 4)):
            members[field] = tuple(item[f"{field}_{i}"] for i in range(count)) if f"{field}_0" in item else None
        return MemberHalf(entry["source"], entry["match"], entry["view"], entry["index"], item["labels"], members)


def collect(cache, identity, produce):
    """Members from `cache` when its identity matches, otherwise computed (as MemberHalf) and cached."""
    manifest = cache / "manifest.json"
    if manifest.exists():
        stored = read(manifest)
        if stored["identity"] != identity:
            raise ValueError(f"{cache} holds members for different models or halves")
        return [load_member(cache, entry) for entry in stored["halves"]]
    halves = list(produce())
    cache.mkdir(parents=True, exist_ok=False)
    entries = [save_member(cache, half) for half in halves]
    manifest.write_text(json.dumps({"identity": identity, "halves": entries}, indent=2, sort_keys=True) + "\n")
    return halves


def as_member(half):
    """A `diagnosis.HalfSignals` whose `signals` hold members (from `MemberForecaster.member_fn`) as a MemberHalf."""
    return MemberHalf(half.source, half.match, half.view, half.index, np.asarray(half.labels), half.signals)


# -- the halves, in the stack's scoring order ---------------------------------------------------------------

def development_halves(dfl, skill, parsed, games, forecast):
    from dfl_submission import diagnosis
    for half in diagnosis.external_halves(dfl, skill, "development", None, raw=True, signals_of=forecast):
        yield as_member(half)
    for half in diagnosis.pff_halves(parsed, games, None, signals_of=forecast):
        yield as_member(half)


def metrica_halves(forecast):
    """As `metrica_sample.reference_halves` makes them."""
    from dfl_submission import metrica_sample as ms
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            members, _ = forecast(None, *ms.tracking(half))
            yield MemberHalf("metrica", str(game), "observed", index, np.asarray(half["labels"]), members)
        print(f"D59 members: Metrica game {game}", flush=True)


def d24_halves(forecast):
    """As `metrica_views.view_halves` makes them."""
    from dfl_submission import metrica_sample as ms, metrica_views as mv
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            for view, (positions, ball) in mv.build_views(half, game, index).items():
                members, _ = forecast(None, positions, ball, half["teams"], half["pitch"])
                yield MemberHalf("metrica", str(game), view, index, np.asarray(half["labels"]), members)
        print(f"D59 members: D24 views of Metrica game {game}", flush=True)


def guard_halves(tracking, forecast):
    """As D48's development makes D49's two long-hidden guard views."""
    from dfl_submission import hidden_views as hv
    for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
        members, _ = forecast(None, positions, ball, teams, pitch)
        yield MemberHalf(source, match, hv.VIEW, index, np.asarray(labels), members)
    print("D59 members: guard views", flush=True)


def situation_halves(tracking, forecast):
    """D55's (d) and (a) views of WPY, as D56 scores them."""
    for (name, i), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
        members, _ = forecast(None, positions, ball, teams, pitch)
        yield MemberHalf("dfl", "WPY", name, i, np.asarray(labels), members)
    print("D59 members: D55's views", flush=True)


# -- the adopted stack -----------------------------------------------------------------------------------

def adopted():
    """(config, sign map, models, identity) of the adopted stack: the twelve networks in `stack_models/`, checked against
    the confirmed identity and against the identity D48's development (`development_t.json`) scored."""
    from dfl_submission import scale_trial
    confirmation = read(RECIPE / "confirmation.json")
    development = read(RECIPE / "development_t.json")["identity"]["stack"]
    config = dict(confirmation["levers"]["config"])
    networks = confirmation["identity"]["networks"]
    if networks != development["networks"] or config != development["config"] or \
            confirmation["identity"]["trees"] != development["trees"]:
        raise ValueError("The confirmed stack is not the one D48's development scored")
    for offset in st.OFFSETS:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                if digest(st.stack_model_folder(offset, model_set, folder) / "fit.json") != networks[str(offset)][model_set][folder]:
                    raise ValueError(f"The adopted network {offset}/{model_set}/{folder} differs from the confirmed one")
    for key in st.SETS:
        if digest(scale_trial.TREES[key] / "fit.json") != development["trees"][key]:
            raise ValueError("D3's trees differ from the confirmed ones")
    sign_map = read(st.PARITY)["sign_map"] if config["mirror"] is not None else None
    models = st.stack_models(config, st.MODELS, export=st.stack_model_folder, sign_map=sign_map, names=("stack",))
    return config, sign_map, models, {"networks": networks, "config": config, "trees": development["trees"],
                                      "code": {name: digest(PACKAGE / name) for name in ("stack_set.py", "player_set.py",
                                                                                         "direction_tokens.py", "player_set3.py",
                                                                                         "temporal_cpu.py")}}


# -- the member pass ---------------------------------------------------------------------------------------

def member_pass(dfl, skill, parsed, *, cache=MEMBERS):
    """One pass of the adopted stack over every scored half, D3's and R's trees side by side, cached; then the members
    re-blended without H are checked against the stack's cached decoder inputs, bit for bit."""
    require_external_only()
    from dfl_submission import competition_adaptation
    st.wait_for_scoring("D59 member pass")
    parts, forecaster = _member_pass(dfl, skill, parsed, cache=cache)
    reference_model, _, frozen = competition_adaptation.reference()
    bits = check_against_stack(parts, reference_model[2]["recipe"], frozen["recipe"])
    print(f"D59 members: {sum(len(h) for h in parts.values())} halves; signs used {len(forecaster.signs)}, fallbacks "
          f"{sorted({row['half'] for row in forecaster.signs if row['fallback']})}; re-blended without H equal to the "
          f"stack's cached inputs: {bits}", flush=True)
    return bits


def inputs_identity(skill, parsed, games):
    from dfl_submission import metrica_sample as ms, vendor_trial
    return {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
            "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": True}


def _member_pass(dfl, skill, parsed, *, cache):
    from dfl_submission import diagnosis, hidden_trial as h49, situation_fit as sf, tree_share_trial as tt, vendor_trial
    config, _, models, stack_identity = adopted()
    refit = tt.refit_folders()
    games = vendor_trial.development()
    situation = sf.situation_tracking(dfl)
    situation_names = {("air_or_crowd", i) for i in (0, 1)} | {("air", i) for i in (0, 1)}
    situation = {key: value for key, value in situation.items() if key in situation_names}
    table = st.table_of(st.SCRATCH / "sign_table_development.json") | sf.situation_table(situation)
    forecaster = MemberForecaster(models, table, refit=refit)          # models: {"stack": {role: ...}}
    forecast = forecaster.member_fn()
    identity = {"protocol": PROTOCOL, "stack": stack_identity,
                "refit": {key: digest(folder / "fit.json") for key, folder in refit.items()},
                "refit_record_sha256": digest(RECIPE / "d59_refit.json"), "teacher_sha256": diagnosis.teacher_identity(),
                "sign_table_sha256": digest(st.SCRATCH / "sign_table_development.json"),
                "inputs": inputs_identity(skill, parsed, games), "guard": h49.guard_identity({"model": "stack"}, raw=True),
                "situation": sf.situation_identity(), "tree_share_sha256": digest(PACKAGE / "tree_share.py")}
    guards = h49.guard_tracking(dfl, raw=True)
    producers = {"development": lambda: development_halves(dfl, skill, parsed, games, forecast),
                 "metrica": lambda: metrica_halves(forecast), "d24": lambda: d24_halves(forecast),
                 "guards": lambda: guard_halves(guards, forecast), "situation": lambda: situation_halves(situation, forecast)}
    parts = {part: collect(cache / part, dict(identity, part=part), producers[part]) for part in PARTS}
    return parts, forecaster


def check_against_stack(parts, previous, recipe, stack_cache=STACK_CACHE):
    """With D3's trees and no H, every half's blended inputs equal the adopted stack's cached ones, bit for bit."""
    from dfl_submission import diagnosis
    result = {}
    for part in ("development", "metrica", "d24", "guards"):
        stored = read(stack_cache / part / "manifest.json")["halves"]
        halves = parts[part]
        same_order = [(e["source"], e["match"], e["view"], e["index"]) for e in stored] == \
            [(h.source, h.match, h.view, h.index) for h in halves]
        equal = same_order
        if same_order:
            for entry, half in zip(stored, halves, strict=True):
                cached = diagnosis.load_half(stack_cache / part, entry)
                fresh = signals(half.members, "d3", None, previous, recipe)
                equal = equal and all(np.array_equal(a, b) for a, b in zip(fresh, cached.signals, strict=True)) \
                    and np.array_equal(half.labels, cached.labels)
        result[part] = {"halves": len(halves), "same_halves_in_order": bool(same_order), "bit_for_bit": bool(equal)}
    return result


# -- scoring -----------------------------------------------------------------------------------------------

def scored(parts, tree_set, share, previous, recipe, frozen_recipe):
    """{part: [HalfSignals]} of one candidate: its blended inputs, decoded at the frozen recipe as the forecaster does."""
    from dfl_submission import diagnosis
    result = {}
    for part, halves in parts.items():
        result[part] = []
        for half in halves:
            blended = signals(half.members, tree_set, share, previous, recipe)
            result[part].append(diagnosis.HalfSignals(half.source, half.match, half.view, half.index, half.labels, blended,
                                                      diagnosis.decode(blended, frozen_recipe)))
    return result


def entries_for(halves, costs, truths, visibility, situation_tracking):
    """[(full entry, compact item)] at each cost: D47's groups, D49's guards, D33's conventions, D55's views."""
    from dfl_submission import hidden_trial as h49, hyper_trial as ht, situation_fit as sf
    result = []
    for cost in costs:
        full = h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"], ht.recipe_at(cost),
                         truths, visibility)
        item = dict(h49.compact(full), selection=st.selection(full))
        item |= sf.situation_scores(halves["situation"], situation_tracking, cost)
        result.append((full, item))
    return result


def conventions(trial):
    from dfl_submission import refine_trial
    return {name: refine_trial.standard(trial, name) for name in (refine_trial.PRIMARY, *refine_trial.OTHER_CONVENTIONS)}


def judge(item, full, reference, reference_full):
    """D59's five conditions for one candidate at one cost against the adopted stack at cost 14."""
    from dfl_submission import refine_trial
    groups = {name: item["groups"][name] - reference["groups"][name] for name in st.GROUPS}
    named = {name: refine_trial.standard(full["trial"], name) - refine_trial.standard(reference_full["trial"], name)
             for name in refine_trial.OTHER_CONVENTIONS}
    changes = {"selection": item["selection"] - reference["selection"], "pff": groups["pff"],
               "composite": item["composite"] - reference["composite"], "groups": groups, "conventions": named,
               "guard_mean": item["guards"]["mean"] - reference["guards"]["mean"],
               "primary": item["primary"] - reference["primary"],
               "situation_d": item["situation_d"] - reference["situation_d"],
               "pff_unseen_agreement": item["pff_visibility"]["agreement_unseen"]["agreement"]
               - reference["pff_visibility"]["agreement_unseen"]["agreement"]}
    conditions = {"selection": bool(changes["selection"] >= SELECTION_GAIN), "pff": bool(changes["pff"] >= PFF_GAIN),
                  "composite": bool(changes["composite"] >= 0),
                  "groups": bool(all(value >= -LOSS for value in groups.values())),
                  "conventions": bool(all(value >= -LOSS for value in named.values()))}
    return {"owner_cost": item["owner_cost"], "changes": changes, "conditions": conditions,
            "holds": bool(all(conditions.values()))}


def best(items):
    """The cost with the highest selection score, ties to the lower cost (D48's gate)."""
    return max(items, key=lambda pair: (pair[1]["selection"], -pair[1]["owner_cost"]))


def decision(results):
    passing = [name for name, result in results.items() if result["holds"]]
    return max(passing, key=lambda name: results[name]["selection"]) if passing else None


def score(dfl, skill, parsed, *, cache=MEMBERS, output=None):
    """Pipeline check 1, the reference and every candidate at each cost, D59's rule; records written once."""
    require_external_only()
    from dfl_submission import competition_adaptation, situation_fit as sf
    output = RECIPE / "d59_scores.json" if output is None else output
    decision_path = output.parent / "d59_decision.json"
    for path in (output, decision_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; D59 records are written once")
    st.wait_for_scoring("D59 scoring")
    parts, _ = _member_pass(dfl, skill, parsed, cache=cache)
    reference_model, _, frozen = competition_adaptation.reference()
    previous, recipe = reference_model[2]["recipe"], frozen["recipe"]
    games, truths, visibility = sf.development_inputs(dfl, skill, parsed)
    tracking = {key: value for key, value in sf.situation_tracking(dfl).items() if key[0] in sf.VIEWS.values()}
    bits = check_against_stack(parts, previous, recipe)
    print(f"D59: the members re-blended without H equal the stack's cached inputs: {bits}", flush=True)
    variants = {"reference": REFERENCE} | CANDIDATES
    entries, items = {}, {}
    for name, (tree_set, share) in variants.items():
        halves = scored(parts, tree_set, share, previous, recipe, recipe)
        pairs = entries_for(halves, COSTS, truths, visibility, tracking)
        entries[name], items[name] = [full for full, _ in pairs], [item for _, item in pairs]
        for item in items[name]:
            print(f"D59 {name} cost {item['owner_cost']}: selection {item['selection']:.5f}, composite {item['composite']:.5f}, "
                  f"guards {item['guards']['mean']:.4f}, PFF {item['groups']['pff']:.4f}, (d) {item['situation_d']:.4f}",
                  flush=True)
    at = {item["owner_cost"]: index for index, item in enumerate(items["reference"])}
    reference, reference_full = items["reference"][at[REFERENCE_COST]], entries["reference"][at[REFERENCE_COST]]
    stored = next(item for item in read(RECIPE / "development_t.json")["entries"]["stack"] if item["owner_cost"] == REFERENCE_COST)
    mine = {key: value for key, value in reference.items() if key not in ("situation_d", "situation_a")}
    d55 = read(RECIPE / "situation_views.json")["scores"]["stack"]["air_or_crowd"]
    check1 = {"cost_14_row_equal": canonical(mine) == canonical(stored),
              "differing_keys": sorted(key for key in stored if canonical(stored[key]) != canonical(mine.get(key))),
              "situation_d_equals_d55_record": bool(d55["owner_cost"] == REFERENCE_COST
                                                    and d55["spell_f1"] == reference["situation_d"])}
    passed = bool(check1["cost_14_row_equal"] and all(part["bit_for_bit"] for part in bits.values()))
    results, per_cost = {}, {}
    for name in CANDIDATES:
        pairs = list(zip(entries[name], items[name]))
        per_cost[name] = {str(item["owner_cost"]): judge(item, full, reference, reference_full) for full, item in pairs}
        full, item = best(pairs)
        results[name] = dict(judge(item, full, reference, reference_full), selection=item["selection"])
    taken = decision(results) if passed else None
    write_new(output, canonical({
        "protocol": PROTOCOL, "passed_checks": passed, "check_1": check1, "members_bit_for_bit": bits,
        "costs": list(COSTS), "reference": {"name": "the adopted stack", "owner_cost": REFERENCE_COST, **reference},
        "variants": {name: {"trees": tree_set, "unseen_tree_share": share} for name, (tree_set, share) in variants.items()},
        "entries": items, "conventions": {name: [conventions(full["trial"]) for full in entries[name]] for name in variants},
        "grid": "the half's 25 Hz frames; unseen = the forecast's ball has no x or y at the frame",
        "members": {part: read(cache / part / "manifest.json")["identity"] for part in PARTS},
        "member_caches": {part: digest(cache / part / "manifest.json") for part in PARTS},
        "refit_sha256": digest(RECIPE / "d59_refit.json"), "development_t_sha256": digest(RECIPE / "development_t.json"),
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    write_new(decision_path, canonical({
        "protocol": PROTOCOL, "passed_checks": passed, "results": results, "per_cost": per_cost, "taken": taken,
        "rule": {"selection_gain": SELECTION_GAIN, "pff_gain": PFF_GAIN, "composite": "not lower", "group_loss": LOSS,
                 "conventions_loss": LOSS, "against": "the adopted stack at cost 14",
                 "cost": "each candidate at its cost with the highest selection score (ties to the lower), as D48's gate"},
        "scores_sha256": digest(output), "decided_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    print(summary(items), flush=True)
    for name, result in results.items():
        changes = result["changes"]
        print(f"D59 {name}: cost {result['owner_cost']}, selection {changes['selection']:+.4f} (+{SELECTION_GAIN}), PFF "
              f"{changes['pff']:+.4f} (+{PFF_GAIN}), composite {changes['composite']:+.4f} (>= 0), worst group "
              f"{min(changes['groups'].values()):+.4f}, worst convention {min(changes['conventions'].values()):+.4f} "
              f"(>= -{LOSS}) -> {'holds' if result['holds'] else 'does not hold'}", flush=True)
    print(f"D59 checks passed: {passed}; decision: {taken if taken else 'none holds; the adopted stack stands'}", flush=True)
    return taken


SHORT = {"dfl_clean": "clean", "dfl_masked": "masked", "dfl_dead_carried": "dead carried", "dfl_team_swap": "swap",
         "skillcorner": "SkillCorner", "pff": "PFF", "metrica": "Metrica", "d24": "D24"}


def summary(items):
    lines = ["| Model | Cost | Selection | Composite | Guard mean | Guard DFL | Guard Metrica | PFF | Primary | "
             + " | ".join(SHORT[g] for g in st.GROUPS) + " | (d) view | PFF unseen agreement | PFF <20%-seen recall |",
             "|---|---:|" + "---:|" * (10 + len(st.GROUPS))]
    for name, rows in items.items():
        for item in rows:
            visibility = item["pff_visibility"]
            lines.append(f"| {name} | {item['owner_cost']} | {item['selection']:.5f} | {item['composite']:.5f} | "
                         f"{item['guards']['mean']:.4f} | {item['guards']['a_dfl']:.4f} | {item['guards']['b_metrica']:.4f} | "
                         f"{item['groups']['pff']:.4f} | {item['primary']:.5f} | "
                         + " | ".join(f"{item['groups'][g]:.4f}" for g in st.GROUPS)
                         + f" | {item['situation_d']:.4f} | {visibility['agreement_unseen']['agreement']:.4f} | "
                         f"{visibility['under_20%_seen_spells']['recall_iou0.5']:.4f} |")
    return "\n".join(lines)
