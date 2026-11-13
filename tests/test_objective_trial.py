"""D33 plumbing: the grid, folder mapping, recipes and selection (dfl_submission/objective_trial.py)."""

from dfl_submission import objective_trial as ot, scale_trial


def test_grid_has_36_points_in_objective_then_decoder_order():
    points = ot.points()
    assert len(points) == 36
    assert points[0] == {"alpha": 0., "edge": .2, "owner_cost": 8, "discount": .8}
    assert points[1] == {"alpha": 0., "edge": .2, "owner_cost": 8, "discount": .9}
    assert points[6]["edge"] == .6 and points[-1] == {"alpha": 1., "edge": .6, "owner_cost": 12, "discount": .9}


def test_names_are_stable():
    assert [ot.name(o) for o in ot.OBJECTIVES] == ["a0_e0.2", "a0_e0.6", "a0.5_e0.2", "a0.5_e0.6", "a1_e0.2", "a1_e0.6"]


def test_reference_objective_uses_d15s_published_networks():
    assert ot.network_folders(ot.REFERENCE) == scale_trial.network_folders((96, 5400))
    folders = ot.network_folders((.5, .6))
    assert folders == {key: ot.MODELS / "a0.5_e0.6" / key for key in ot.SETS}


def test_point_recipe_changes_only_cost_and_discount():
    _, frozen = ot.d15()
    base = frozen["recipe"]
    recipe = ot.point_recipe({"alpha": 1., "edge": .6, "owner_cost": 12, "discount": .9})
    assert recipe["owner_cost"] == 12 and recipe["discount"] == .9
    assert {k: v for k, v in recipe.items() if k not in ("owner_cost", "discount")} == \
        {k: v for k, v in base.items() if k not in ("owner_cost", "discount")}
    assert ot.point_recipe({"alpha": 0., "edge": .2, "owner_cost": 8, "discount": .8}) == base


def trial(primary, macro=.5):
    return {"external": {"conventions": {"aba3": {"standard": {"spell_f1": primary, "macro_f1": macro}}}}}


def test_select_takes_the_best_eligible_then_macro_then_grid_order():
    points = ot.points()[:4]
    entries = [{"point": points[0], "eligible": True, "trial": trial(.73)},
               {"point": points[1], "eligible": False, "trial": trial(.80)},
               {"point": points[2], "eligible": True, "trial": trial(.74, .5)},
               {"point": points[3], "eligible": True, "trial": trial(.74, .5)}]
    assert ot.select(entries) == points[2]
    entries[3]["trial"] = trial(.74, .6)
    assert ot.select(entries) == points[3]
    assert ot.select([dict(entry, eligible=False) for entry in entries]) is None


def test_eligibility_needs_the_gain_and_both_vetoes():
    change = {"primary": .012, "conventions": {"aba1.75": -.005, "aba4": 0., "aba4.5": .01},
              "guards": {"dead_carried": -.009, "team_swap": .02}, "pff": -.0099}
    assert ot.eligible(change, True, -.0099)
    assert not ot.eligible(change, False, 0.)
    assert not ot.eligible(change, True, -.011)
    assert not ot.eligible(dict(change, primary=.0099), True, 0.)
    assert not ot.eligible(dict(change, pff=-.011), True, 0.)
