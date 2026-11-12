"""D11 and D21: where frozen D3, then D15, loses spellF1 on the external development matches.

`measure` decodes frozen D3 on DFL WPY (four views), the four SkillCorner development matches and
the eight PFF development matches, checks that it reproduces the D3 and D7 records exactly, and
writes one record: spell detection per IoU, two oracle runs, signed boundary offsets by transition
type, recall by spell length, brief interior spells and stoppage lengths. It selects nothing. The
record also fixes, by the formula in D11, the boundary shifts D12 may test.

The decoder inputs of every half can be cached (`--cache`), so later trials on the same matches
reuse them instead of running the model sets again; each later trial still checks that the cached
inputs reproduce D3.

`measure-d15` (DECISIONS.md D21) repeats the measurements for frozen D15 from its cached development
inputs and adds the three Metrica sample matches as a seventh group, with D3 measured there too.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dfl_submission import pff
from dfl_submission.conventions import live_segments
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


PACKAGE = Path(__file__).resolve().parent
RECIPE = PACKAGE / "diagnosis_recipe"
PROTOCOL = "DECISIONS.md D11"
FPS = 25
TARGETS = {"raw": None, "aba3": 3.0}
TOLERANCE = 25
TYPES = ("ownership", "stop", "restart")
LENGTH_BINS = (0, 1.75, 3, 5, 10, 20, np.inf)
BRIEF = (1.75, 3.0)
HEADLINE_IOU = .5
MIN_SHIFT = 2
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff")
VENDORS = {"dfl": ("dfl_clean", "dfl_masked"), "skillcorner": ("skillcorner",), "pff": ("pff",)}
D21_PROTOCOL = "DECISIONS.md D21"
D21_GROUPS = (*GROUPS, "metrica")
D21_VENDORS = VENDORS | {"metrica": ("metrica",)}


@dataclass
class HalfSignals:
    source: str
    match: str
    view: str
    index: int
    labels: np.ndarray
    signals: tuple
    d3: np.ndarray

    @property
    def key(self):
        return self.source, self.match, self.view

    @property
    def group(self):
        return self.source if self.source != "dfl" else f"dfl_{self.view}"


# -- decoder inputs --------------------------------------------------------------------------------

def half_signals(teacher, positions, ball, teams, pitch):
    """D3's four decoder inputs for one half and its decoded labels, exactly as the build makes them."""
    from dfl_submission import motion_trial, retrain_trial, vendor_trial
    reference, _, frozen = teacher
    previous = reference[2]["recipe"]
    mixed = retrain_trial.mix(*vendor_trial.model_outputs(teacher, positions, ball, teams, pitch), frozen["weight"])
    labels = motion_trial.decode(mixed, previous, frozen["recipe"])
    signals = tuple(motion_trial.blend_signals(mixed, previous, frozen["recipe"]))
    if not np.array_equal(decode(signals, frozen["recipe"]), labels):
        raise ValueError("Decoding the cached inputs does not reproduce D3")
    return signals, labels


def decode(signals, recipe, **options):
    from dfl_submission import joint_decoder
    return joint_decoder.decode(*signals, cutoff=recipe["cutoff"], dead_cost=recipe["dead_cost"],
                                owner_cost=recipe["owner_cost"], discount=recipe["discount"], **options)


def cache_name(source, match, view, index):
    return f"{source}_{match}_{view}_{index}.npz"


def save_half(folder, half):
    path = folder / cache_name(half.source, half.match, half.view, half.index)
    np.savez_compressed(path, labels=half.labels, d3=half.d3, home=half.signals[0], dead=half.signals[1],
                        dead_edge=half.signals[2], owner_edge=half.signals[3])
    return {"source": half.source, "match": half.match, "view": half.view, "index": half.index,
            "file": path.name, "sha256": digest(path)}


def load_half(folder, entry):
    path = folder / entry["file"]
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Cached decoder inputs differ from the manifest: {path.name}")
    with np.load(path, allow_pickle=False) as item:
        signals = tuple(item[name] for name in ("home", "dead", "dead_edge", "owner_edge"))
        return HalfSignals(entry["source"], entry["match"], entry["view"], entry["index"], item["labels"], signals,
                           item["d3"])


def external_halves(dfl, skill, split, teacher, *, raw, signals_of=half_signals):
    from dfl_submission import augment
    for half in augment.guard_halves(dfl, skill, split, raw=raw):
        signals, labels = signals_of(teacher, half.positions, half.ball, half.teams, half.pitch)
        yield HalfSignals(half.source, half.match, half.view, half.index, np.asarray(half.labels), signals, labels)
        print(f"Forecast {half.key}", flush=True)


def pff_halves(parsed, games, teacher, *, signals_of=half_signals):
    for game in games:
        for index, half in enumerate(pff.load(parsed, game)):
            signals, labels = signals_of(teacher, *pff.tracking(half))
            yield HalfSignals("pff", str(game), "observed", index, np.asarray(half["labels"]), signals, labels)
        print(f"Forecast PFF {game}", flush=True)


def collect(cache, identity, produce):
    """Decoder inputs from `cache` when its identity matches, otherwise computed and cached."""
    manifest = None if cache is None else cache / "manifest.json"
    if manifest is not None and manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["identity"] != identity:
            raise ValueError(f"{cache} holds decoder inputs for different models or matches")
        return [load_half(cache, entry) for entry in stored["halves"]]
    halves = list(produce())
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=False)
        entries = [save_half(cache, half) for half in halves]
        manifest.write_text(json.dumps({"identity": identity, "halves": entries}, indent=2, sort_keys=True) + "\n")
    return halves


def teacher_identity():
    from dfl_submission import competition_adaptation
    return digest(competition_adaptation.BLEND / "recipe.json")


def development_halves(dfl, skill, parsed, teacher, *, raw, cache=None):
    """DFL/SkillCorner development halves with the D2 guard views, then the PFF development halves."""
    from dfl_submission import vendor_trial
    games = vendor_trial.development()
    identity = {"teacher_sha256": teacher_identity(), "split": "development", "dfl_raw": raw,
                "skill_manifest_sha256": digest(skill / "manifest.json"),
                "pff": vendor_trial.inputs(parsed, games)}

    def produce():
        yield from external_halves(dfl, skill, "development", teacher, raw=raw)
        yield from pff_halves(parsed, games, teacher)

    return collect(cache, identity, produce)


def matches(halves, predict=lambda half: half.d3):
    """{key: (truth, prediction, home)} over both halves, with an X frame between them."""
    grouped = {}
    for half in halves:
        grouped.setdefault(half.key, []).append(half)
    result = {}
    for key, parts in grouped.items():
        if [part.index for part in parts] != [0, 1]:
            raise ValueError(f"Expected two halves in order for {key}")
        truth = np.r_[parts[0].labels, pm.X, parts[1].labels].astype(np.int8)
        prediction = np.r_[predict(parts[0]), pm.X, predict(parts[1])].astype(np.int8)
        home = np.r_[parts[0].signals[0], np.nan, parts[1].signals[0]]
        result[key] = (truth, prediction, home)
    return result


# -- the D3 reproduction check ---------------------------------------------------------------------

def target(truth, seconds):
    from dfl_submission.decoder_trial import relabel
    return relabel(truth, seconds)


def score(truth, prediction):
    spells = pm.spell_scores(truth, prediction, pm.valid_runs(truth))
    confusion = pm.scored_confusion(pm.confusion(truth, prediction))
    return {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion), "spells": spells["n_pred"]}


def external_record(scored, recipe):
    """The structure of `retrain_trial.evaluate`, from already decoded matches."""
    from dfl_submission import decoder_trial, retrain_trial
    conventions = {}
    for name, seconds in decoder_trial.CONVENTIONS.items():
        conventions[name] = retrain_trial.summarise(
            {key: score(target(truth, seconds), prediction) for key, (truth, prediction, _) in scored.items()})
    return {"recipe": recipe, "conventions": conventions}


def pff_record(scored):
    """The per-game structure of the D7 measurement, from already decoded matches."""
    from dfl_submission import decoder_trial
    result = {}
    for (_, game, _), (truth, prediction, _) in scored.items():
        result[game] = {}
        for name, seconds in decoder_trial.CONVENTIONS.items():
            result[game][name] = score(target(truth, seconds), prediction)
    return result


def split_sources(scored):
    external = {key: value for key, value in scored.items() if key[0] != "pff"}
    vendor = {key: value for key, value in scored.items() if key[0] == "pff"}
    return external, vendor


def reproduces(scored, recipe):
    """Whether D3 decoded from these inputs reproduces the D3 development record and the D7 measurement."""
    from dfl_submission import vendor_trial
    external, vendor = split_sources(scored)
    return vendor_trial.reproduction(pff_record(vendor), external_record(external, recipe))


# -- measurements ----------------------------------------------------------------------------------

def spell_counts(truth, prediction):
    spells = pm.spell_scores(truth, prediction, pm.valid_runs(truth))
    return {str(iou): {key: spells["per_iou"][iou][key] for key in ("n_matched", "n_spurious", "n_missed")}
            for iou in pm.SPELL_IOUS} | {"n_gt": spells["n_gt"], "n_pred": spells["n_pred"],
                                         "spell_f1": spells["mean_f1"]}


def nearest_fill(values, keep):
    """Replace values where `keep` is False by the nearest kept value, the earlier one on ties."""
    index = np.arange(len(values))
    before = np.maximum.accumulate(np.where(keep, index, -1))
    after = np.minimum.accumulate(np.where(keep, index, len(values))[::-1])[::-1]
    use_before = (before >= 0) & ((after >= len(values)) | (index - before <= after - index))
    source = np.where(use_before, before, after)
    return values[source]


def oracle_stoppage(truth, prediction, home):
    """True stoppages substituted into D3 (DECISIONS.md D11)."""
    result = prediction.copy()
    result[truth == pm.D] = pm.D
    for start, stop in live_segments(truth):
        part = prediction[start:stop]
        live = part != pm.D
        if live.all():
            continue
        if not live.any():
            result[start:stop] = pm.H if np.mean(home[start:stop]) >= .5 else pm.A
        else:
            result[start:stop] = nearest_fill(part, live)
    return result


def oracle_ownership(truth, prediction):
    """True ownership substituted into D3 (DECISIONS.md D11)."""
    result = prediction.copy()
    for start, stop in pm.valid_runs(truth):
        part = truth[start:stop]
        live = part != pm.D
        if not live.any():
            continue
        index = np.arange(len(part))
        before = np.maximum.accumulate(np.where(live, index, -1))
        after = np.minimum.accumulate(np.where(live, index, len(part))[::-1])[::-1]
        team = part[np.where(before >= 0, before, after)]
        window = result[start:stop]
        result[start:stop] = np.where(window == pm.D, pm.D, team)
    return result


def transition_type(origin, destination):
    if destination == pm.D:
        return "stop"
    if origin == pm.D:
        return "restart"
    return "ownership"


def offsets(truth, prediction):
    """Signed offsets (predicted minus true frame) of matched transitions, with missed and spurious counts."""
    runs = pm.valid_runs(truth)
    pairs, missed, spurious = pm.match_transitions(pm.transitions(truth, runs), pm.transitions(prediction, runs),
                                                   TOLERANCE, typed="pair")
    result = {kind: {"offsets": [], "missed": 0, "spurious": 0} for kind in TYPES}
    for gt, predicted in pairs:
        result[transition_type(gt[1], gt[2])]["offsets"].append(predicted[0] - gt[0])
    for name, items in (("missed", missed), ("spurious", spurious)):
        for _, origin, destination in items:
            result[transition_type(origin, destination)][name] += 1
    return result


def matched_flags(overlaps, n_gt, n_pred, threshold):
    """The metric's greedy one-to-one spell match, returned for both sides."""
    gt, predicted = np.zeros(n_gt, dtype=bool), np.zeros(n_pred, dtype=bool)
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if gt[i] or predicted[j]:
            continue
        gt[i] = predicted[j] = True
    return gt, predicted


def spell_lengths(truth, prediction):
    """Per true spell: length in frames and matched flags at each IoU; per predicted interior spell: the same."""
    runs = pm.valid_runs(truth)
    true_spells, predicted_spells = pm.possession_spells(truth, runs), pm.possession_spells(prediction, runs)
    overlaps = pm.spell_overlaps(true_spells, predicted_spells)
    flags = {iou: matched_flags(overlaps, len(true_spells), len(predicted_spells), iou) for iou in pm.SPELL_IOUS}
    true_lengths = np.array([stop - start for start, stop, _ in true_spells], dtype=np.int64)
    true_matched = np.array([flags[iou][0] for iou in pm.SPELL_IOUS], dtype=bool).reshape(len(pm.SPELL_IOUS), -1)
    segments = pm.segments(prediction, runs)
    interior = set()
    for i in range(1, len(segments) - 1):
        before, middle, after = segments[i - 1], segments[i], segments[i + 1]
        if (before[1] == middle[0] and middle[1] == after[0] and middle[2] in pm.POSSESSION_LABELS
                and before[2] == after[2] and before[2] in pm.POSSESSION_LABELS):
            interior.add((middle[0], middle[1]))
    position = {(start, stop): j for j, (start, stop, _) in enumerate(predicted_spells)}
    brief = [(stop - start, bool(flags[HEADLINE_IOU][1][position[start, stop]])) for start, stop in sorted(interior)]
    return true_lengths, true_matched, brief


def stoppages(labels, runs):
    return np.array([stop - start for start, stop, code in pm.segments(labels, runs) if code == pm.D], dtype=np.int64)


def length_summary(lengths):
    if not len(lengths):
        return {"count": 0}
    seconds = lengths / FPS
    return {"count": int(len(seconds)), "p10_s": float(np.percentile(seconds, 10)),
            "p50_s": float(np.percentile(seconds, 50)), "p90_s": float(np.percentile(seconds, 90)),
            "under_1s": int(np.sum(seconds < 1)), "under_3s": int(np.sum(seconds < 3))}


def offset_summary(items):
    values = np.array([value for item in items for value in item["offsets"]], dtype=np.float64)
    summary = {"matched": int(len(values)), "missed": int(sum(item["missed"] for item in items)),
               "spurious": int(sum(item["spurious"] for item in items))}
    if len(values):
        summary |= {"median": float(np.median(values)), "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)), "mean": float(values.mean())}
    return summary


def measure_match(truth, prediction, home):
    runs = pm.valid_runs(truth)
    true_lengths, true_matched, brief = spell_lengths(truth, prediction)
    return {"spells": spell_counts(truth, prediction),
            "oracle": {"stoppage": score(truth, oracle_stoppage(truth, prediction, home))["spell_f1"],
                       "ownership": score(truth, oracle_ownership(truth, prediction))["spell_f1"]},
            "offsets": offsets(truth, prediction),
            "true_lengths": true_lengths, "true_matched": true_matched, "brief": brief,
            "stoppages": {"predicted": stoppages(prediction, runs), "true": stoppages(truth, runs)}}


def pooled_spells(records):
    result = {}
    for iou in map(str, pm.SPELL_IOUS):
        tp, fp, fn = (sum(record["spells"][iou][key] for record in records)
                      for key in ("n_matched", "n_spurious", "n_missed"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[iou] = {"precision": precision, "recall": recall,
                       "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return result | {"n_gt": sum(record["spells"]["n_gt"] for record in records),
                     "n_pred": sum(record["spells"]["n_pred"] for record in records)}


def length_recall(records):
    lengths = np.concatenate([record["true_lengths"] for record in records]) / FPS
    matched = np.concatenate([record["true_matched"] for record in records], axis=1)
    result = {}
    for low, high in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
        inside = (lengths >= low) & (lengths < high)
        label = f"{low:g}-{high:g}s" if np.isfinite(high) else f"{low:g}s+"
        count = int(inside.sum())
        result[label] = {"count": count}
        if count:
            result[label] |= {"recall_iou0.5": float(matched[pm.SPELL_IOUS.index(HEADLINE_IOU), inside].mean()),
                              "recall_mean": float(matched[:, inside].mean())}
    return result


def brief_summary(records):
    items = [item for record in records for item in record["brief"]]
    return {f"under_{limit:g}s": {"count": sum(length < limit * FPS for length, _ in items),
                                  "matched_iou0.5": sum(length < limit * FPS and hit for length, hit in items)}
            for limit in BRIEF}


def summarise_group(records):
    return {"matches": len(records),
            "spell_f1_mean": float(np.mean([record["spells"]["spell_f1"] for record in records])),
            "detection": pooled_spells(records),
            "oracle": {name: float(np.mean([record["oracle"][name] for record in records]))
                       for name in ("stoppage", "ownership")},
            "offsets": {kind: offset_summary([record["offsets"][kind] for record in records]) for kind in TYPES},
            "length_recall": length_recall(records),
            "brief_interior": brief_summary(records),
            "stoppages": {side: length_summary(np.concatenate([record["stoppages"][side] for record in records]))
                          for side in ("predicted", "true")}}


def combined(per_match, metric):
    """The standard combined aggregate (DFL 1/3 clean + 2/3 masked, SkillCorner mean, averaged)."""
    from dfl_submission import context_trial
    standard = {key: {"value": value[metric]} for key, value in per_match.items()
                if key[0] == "skillcorner" or key[2] in ("clean", "masked")}
    return context_trial.aggregate(standard, "value")


def round_half_away(value):
    return int(np.sign(value) * np.floor(abs(value) + .5))


def shifts(vendor_offsets):
    """D11's rule for the boundary shifts D12 may test, from the N = 3 median offsets per vendor."""
    result = {}
    for kind in TYPES:
        medians = {vendor: vendor_offsets[vendor][kind].get("median") for vendor in VENDORS}
        entry = {"medians": medians, "shift": None}
        if all(value is not None for value in medians.values()):
            signs = {int(np.sign(value)) for value in medians.values()}
            size = round_half_away((medians["dfl"] + medians["skillcorner"]) / 2)
            if len(signs) == 1 and 0 not in signs and abs(size) >= MIN_SHIFT:
                entry["shift"] = -size
        result[kind] = entry
    return result


def measure_all(scored, *, groups=GROUPS, vendors=VENDORS, name="d3", with_shifts=True):
    """Every D11 measurement for every group and target; D21 adds the Metrica group and drops the shifts."""
    report = {"groups": {}, "combined": {}, "vendors": {}}
    by_group = {}
    for key in scored:
        source, _, view = key
        by_group.setdefault(source if source != "dfl" else f"dfl_{view}", []).append(key)
    for target_name, seconds in TARGETS.items():
        records = {}
        for key, (truth, prediction, home) in scored.items():
            records[key] = measure_match(target(truth, seconds), prediction, home)
        report["groups"][target_name] = {group: summarise_group([records[key] for key in sorted(by_group[group])])
                                         for group in groups}
        per_match = {key: {name: record["spells"]["spell_f1"], **{f"oracle_{k}": v for k, v in record["oracle"].items()}}
                     for key, record in records.items() if key[0] in ("dfl", "skillcorner")}
        if per_match:
            report["combined"][target_name] = {metric: combined(per_match, metric)
                                               for metric in (name, "oracle_stoppage", "oracle_ownership")}
        report["vendors"][target_name] = {vendor: {kind: offset_summary([records[key]["offsets"][kind]
                                                                         for group in members for key in by_group[group]])
                                                   for kind in TYPES}
                                          for vendor, members in vendors.items()}
    if with_shifts:
        report["shifts"] = shifts(report["vendors"]["aba3"])
    return report


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D11 is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def measure(dfl, skill, parsed, *, raw, cache=None, output=RECIPE / "d11.json"):
    require_external_only()
    from dfl_submission import competition_adaptation, vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D11 is written once")
    games = vendor_trial.development()
    if not vendor_trial.gate_passed(vendor_trial.RECIPE / "sanity.json", vendor_trial.inputs(parsed, games)):
        raise ValueError("The latest reader sanity attempt did not pass on these parsed inputs")
    teacher = competition_adaptation.reference()
    halves = development_halves(dfl, skill, parsed, teacher, raw=raw, cache=cache)
    scored = matches(halves)
    checks = reproduces(scored, teacher[2]["recipe"])
    if not all(checks.values()):
        raise ValueError(f"D11 pipeline check failed {checks}; no record was written")
    report = measure_all(scored)
    write_new(output, {"protocol": PROTOCOL, "candidate": "D3", "decision": None, "reproduction": checks,
                       "tolerance_frames": TOLERANCE, **report, "teacher_sha256": teacher_identity(),
                       "inputs": vendor_trial.inputs(parsed, games)})
    for name in TARGETS:
        values = report["combined"][name]
        print(f"D11 {name}: standard combined D3 {values['d3']['combined']:.4f}, true stoppages "
              f"{values['oracle_stoppage']['combined']:.4f}, true ownership {values['oracle_ownership']['combined']:.4f}")
    print("D11 shifts for D12: " + ", ".join(f"{kind} {entry['shift']}" for kind, entry in report["shifts"].items()))
    return report


def measure_d15(dfl, skill, parsed, *, raw, d15_cache, output=RECIPE / "d21.json"):
    """D21: D11's measurements for frozen D15 on development and the Metrica sample games, written once."""
    require_external_only()
    from dfl_submission import metrica_sample as ms, scale_trial, vendor_trial, zoo
    if output.exists():
        raise FileExistsError(f"{output} exists; D21 is written once")
    games = vendor_trial.development()
    halves, recipe, _ = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    metrica = {}
    for reference in ("D15", "D3"):
        _, cached, _ = ms.reference_scores(reference)  # stops unless the reference reproduces its D19 record
        reference_recipe = ms.reference_recipe(reference)
        decoded = matches(cached, lambda half: decode(half.signals, reference_recipe))
        stored = matches(cached)
        if any(not np.array_equal(decoded[key][1], stored[key][1]) for key in stored):
            raise ValueError(f"{reference} decoded afresh differs from its cached Metrica labels; nothing was written")
        metrica[reference] = decoded
    scored = matches(halves, lambda half: decode(half.signals, recipe)) | metrica["D15"]
    report = measure_all(scored, groups=D21_GROUPS, vendors=D21_VENDORS, name="d15", with_shifts=False)
    d3_metrica = measure_all(metrica["D3"], groups=("metrica",), vendors={"metrica": ("metrica",)}, name="d3",
                             with_shifts=False)
    frozen, setting = scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)
    write_new(output, {"protocol": D21_PROTOCOL, "candidate": "D15", "decision": None, "recipe": recipe,
                       "reproduction": {"d15_development": True, "d15_metrica": True, "d3_metrica": True},
                       "tolerance_frames": TOLERANCE, **report,
                       "d3_metrica": {"groups": d3_metrica["groups"], "vendors": d3_metrica["vendors"]},
                       "models": frozen["models"], "teacher_sha256": teacher_identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)}})
    for name in TARGETS:
        values = report["combined"][name]
        print(f"D21 {name}: standard combined D15 {values['d15']['combined']:.4f}, true stoppages "
              f"{values['oracle_stoppage']['combined']:.4f}, true ownership {values['oracle_ownership']['combined']:.4f}")
        for group in D21_GROUPS:
            entry = report["groups"][name][group]
            print(f"  {group}: spellF1 {entry['spell_f1_mean']:.4f}, true stoppages {entry['oracle']['stoppage']:.4f}, "
                  f"true ownership {entry['oracle']['ownership']:.4f}, spells {entry['detection']['n_gt']} true / "
                  f"{entry['detection']['n_pred']} predicted")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure", "measure-d15"))
    parser.add_argument("--parsed", type=Path, required=True, help="folder written by `dfl_submission.pff prepare`")
    parser.add_argument("--dfl", type=Path, required=True)
    parser.add_argument("--dfl-raw", action="store_true")
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="folder for the decoder inputs of every development half")
    parser.add_argument("--d15-cache", type=Path, help="measure-d15: D15's cached development inputs")
    args = parser.parse_args()
    if args.command == "measure-d15":
        measure_d15(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache)
        return
    measure(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache=args.cache)


if __name__ == "__main__":
    main()
