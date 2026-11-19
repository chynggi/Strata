"""The auto-tuner on a strata-poly run config (Gemma 4, GLM, ... on the engine this fork adds): the family is read from the engine,
only strata-poly's own flags are tried, and no per-request setting is sent. Upstream's families plan as before."""

from __future__ import annotations

import unittest

from strata_control import autotune, presets, tune_knobs

from tests.tuning.stub_session import FakeClock, StubSession, flag

POLY_ARGS = ["--model", "m.gguf", "--max-context", "131072", "--expert-profile", "p.txt", "--draft-model", "d.gguf",
             "--conversation-cache-mib", "auto"]
POLY_FLAGS = ["--max-context", "--threads", "--expert-swaps", "--window", "--lookup", "--expert-profile", "--draft-model",
              "--conversation-cache-mib", "--vram-reserve-mib", "--prompt-batch", "--cpu-share"]
SYSTEM = {"memory": {"total_mib": 64000}, "gpus": [{"index": 0, "vendor": "nvidia", "vram_total_mib": 12288}],
          "cpu": {"logical": 12, "physical": 6, "hybrid": False}}
POLY_IDS = {a.id for a in tune_knobs.ARMS if "poly" in a.families}


def poly_cfg(exe=r"M:\Strata\engine-poly\strata-poly.exe"):
    return {"exe": exe, "args": list(POLY_ARGS), "gpu": 0}


def poly_ctx(**opts):
    return autotune.context_from_config(poly_cfg(), engine_backend="cuda", opts=autotune.Options(**opts),
                                        engine_flags=POLY_FLAGS, system=SYSTEM)


def ids(ctx, mode="quick"):
    return [p.arm.id for p in tune_knobs.plan_arms(ctx, mode=mode)[0]]


class FamilyTest(unittest.TestCase):
    def test_a_strata_poly_config_is_its_own_family(self):
        ctx = poly_ctx()
        self.assertEqual(ctx.family, "poly")
        self.assertEqual((ctx.logical_threads, ctx.physical_threads), (12, 6))
        self.assertEqual(tune_knobs.machine_classes(ctx), ["P"])
        self.assertTrue(tune_knobs.default_unsteady(ctx))
        self.assertTrue(tune_knobs.poly_engine({"exe": "/opt/strata/engine-poly/strata-poly"}))
        self.assertTrue(tune_knobs.poly_engine({"exe": r"C:\x\strata-glm.exe"}))

    def test_upstreams_engine_keeps_its_family(self):
        cfg = {"exe": r"M:\Strata\engine\strata.exe", "args": ["--pack", "P", "--native", "m.gguf", "--mtp", "D"], "gpu": 0}
        ctx = autotune.context_from_config(cfg, engine_backend="cuda", opts=autotune.Options(), system=SYSTEM)
        self.assertEqual(ctx.family, "flash-next")
        self.assertFalse(tune_knobs.default_unsteady(ctx))
        self.assertEqual(presets.choices_from_config(cfg)["family"], "flash-next")
        q36 = {**cfg, "args": ["--native", "m.gguf", "--mtp", "2"]}
        self.assertEqual(presets.choices_from_config(q36)["family"], "qwen36")

    def test_a_preset_made_from_a_strata_poly_config_names_the_family(self):
        self.assertEqual(presets.choices_from_config(poly_cfg())["family"], "poly")

    def test_upstreams_plans_have_no_strata_poly_arm(self):
        for cls_vram in (12000, 32624):
            ctx = tune_knobs.Context(backend="cuda", family="flash-next", vram_mib=cls_vram, ram_gib=64, model_gib=60.0,
                                     context_tokens=131072, logical_threads=12, physical_threads=6, allow_placement=True,
                                     base_args=["--pack", "P", "--native", "m.gguf"])
            self.assertFalse(POLY_IDS & set(ids(ctx, "thorough")))


class PlanTest(unittest.TestCase):
    def test_quick_tries_only_the_exact_thread_count_by_default(self):
        self.assertEqual(ids(poly_ctx()), ["poly_threads_physical"])
        planned = tune_knobs.plan_arms(poly_ctx(), mode="quick")[0][0]
        self.assertEqual(planned.args, {"--threads": 6})
        self.assertEqual(planned.arm.answer, "exact")

    def test_the_small_answer_differences_switch_adds_the_placement_knobs(self):
        self.assertEqual(ids(poly_ctx(allow_placement=True)),
                         ["poly_threads_physical", "poly_expert_swaps_128", "poly_expert_swaps_32", "poly_window_4"])

    def test_the_unmeasured_knobs_are_thorough_only_and_need_the_switch(self):
        new = {"poly_prompt_batch_2048", "poly_prompt_batch_512", "poly_lookup_2", "poly_lookup_5", "poly_cpu_share_0"}
        self.assertFalse(new & set(ids(poly_ctx(allow_placement=True), "quick")))
        self.assertFalse(new & set(ids(poly_ctx(), "thorough")))
        self.assertEqual(new, new & set(ids(poly_ctx(allow_placement=True), "thorough")))
        for a in tune_knobs.ARMS:
            if a.id in new:
                self.assertEqual((a.answer, a.deep_only, a.families), ("placement", True, ("poly",)))

    def test_the_larger_prompt_batch_needs_a_card_with_room(self):
        small = poly_ctx(allow_placement=True)
        small.vram_mib = 8192                       # an 8 GB laptop card: 2048 tokens of batch would take its expert cache
        chosen, skipped = tune_knobs.plan_arms(small, mode="thorough")
        self.assertNotIn("poly_prompt_batch_2048", [p.arm.id for p in chosen])
        self.assertIn("vram ge 12g", {s["id"]: s["reason"] for s in skipped}["poly_prompt_batch_2048"])
        self.assertIn("poly_prompt_batch_512", [p.arm.id for p in chosen])
        big = poly_ctx(allow_placement=True)
        big.vram_mib = 12288
        self.assertIn("poly_prompt_batch_2048", [p.arm.id for p in tune_knobs.plan_arms(big, mode="thorough")[0]])

    def test_a_value_already_set_is_not_tried_again(self):
        ctx = poly_ctx(allow_placement=True)
        ctx.base_args = POLY_ARGS + ["--lookup", "5", "--prompt-batch", "512"]
        got = ids(ctx, "thorough")
        self.assertNotIn("poly_lookup_5", got)
        self.assertNotIn("poly_prompt_batch_512", got)
        self.assertIn("poly_lookup_2", got)

    def test_no_thread_arm_without_hyperthreading_or_when_it_is_already_set(self):
        ctx = poly_ctx()
        ctx.physical_threads = 12
        self.assertNotIn("poly_threads_physical", ids(ctx))
        ctx = poly_ctx()
        ctx.base_args = POLY_ARGS + ["--threads", "6"]
        self.assertNotIn("poly_threads_physical", ids(ctx))

    def test_a_flag_the_engine_does_not_take_is_left_out(self):
        ctx = poly_ctx(allow_placement=True)
        ctx.engine_flags = [f for f in POLY_FLAGS if f != "--window"]
        chosen, skipped = tune_knobs.plan_arms(ctx, mode="quick")
        self.assertNotIn("poly_window_4", [p.arm.id for p in chosen])
        self.assertIn("this engine does not accept --window", {s["id"]: s["reason"] for s in skipped}["poly_window_4"])


class ExpertProfileTest(unittest.TestCase):
    """strata-poly writes --expert-profile back after a run: every measured run starts from the file as it was at the start."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.dir = Path(tempfile.mkdtemp(prefix="strata-poly-tune-test-"))
        self.addCleanup(__import__("shutil").rmtree, self.dir, True)
        self.profile = self.dir / "expert-profile.txt"
        self.profile.write_text("first\n", encoding="utf-8")
        self.args = ["--model", "m.gguf", "--expert-profile", str(self.profile), "--max-context", "8192"]

    def session(self, exe):
        from strata_control.tune_session import WorkerSession
        return WorkerSession(base_config={"exe": exe, "args": list(self.args)}, serve_dir=".", workdir=self.dir / "work")

    def test_each_run_gets_a_copy_of_the_starting_profile(self):
        s = self.session(r"M:\Strata\engine-poly\strata-poly.exe")
        a1 = s._own_expert_profile(list(self.args), 1)
        from pathlib import Path
        p1 = Path(a1[a1.index("--expert-profile") + 1])
        self.assertNotEqual(p1, self.profile)
        p1.write_text("written back by run 1\n", encoding="utf-8")           # what the engine does on exit
        self.profile.write_text("changed elsewhere\n", encoding="utf-8")
        a2 = s._own_expert_profile(list(self.args), 2)
        p2 = Path(a2[a2.index("--expert-profile") + 1])
        self.assertEqual(p2.read_text(encoding="utf-8"), "first\n")
        self.assertEqual(a2[:2] + a2[4:], self.args[:2] + self.args[4:])

    def test_upstreams_engine_and_a_missing_profile_are_left_alone(self):
        self.assertEqual(self.session(r"M:\Strata\engine\strata.exe")._own_expert_profile(list(self.args), 1), self.args)
        self.profile.unlink()
        self.assertEqual(self.session(r"M:\Strata\engine-poly\strata-poly.exe")._own_expert_profile(list(self.args), 1), self.args)


class TunerTest(unittest.TestCase):
    def run_tuner(self, model, ctx):
        clock = FakeClock()
        session = StubSession(model=model, clock=clock, noise=0.002)
        t = autotune.Tuner(session, list(POLY_ARGS), {}, ctx, autotune.Options(), clock=clock, sleep=clock.sleep,
                           system=session.system, total_ram_mib=64000, total_vram_mib=12288, quiet=lambda: None)
        return session, t.run()

    def test_a_faster_thread_count_is_kept_and_no_request_setting_is_sent(self):
        sent = []

        def model(args, env, tune=None):
            if tune:
                sent.append(tune)
            return 60.0 * (1.08 if flag(args, "--threads") == "6" else 1.0)
        session, r = self.run_tuner(model, poly_ctx())
        self.assertEqual(r.outcome, "improved", r.notes)
        self.assertEqual(r.settings, {"args": {"--threads": 6}, "env": {}, "per_request": {}})
        self.assertEqual(r.answers["class"], "exact")
        self.assertFalse(r.answers["default_repeatable"])
        self.assertEqual(sent, [])
        self.assertEqual(r.per_request, [])
        for args, _env in session.runs:                     # the person's own flags stay in every run
            self.assertEqual(flag(args, "--draft-model"), "d.gguf")
            self.assertEqual(flag(args, "--max-context"), "131072")

    def test_no_gain_keeps_the_standard_settings(self):
        _session, r = self.run_tuner(lambda a, e, tune=None: 60.0, poly_ctx())
        self.assertEqual(r.outcome, "default_confirmed")
        self.assertEqual(r.settings, {"args": {}, "env": {}, "per_request": {}})


if __name__ == "__main__":
    unittest.main()
