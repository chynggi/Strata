"""`strata launch` with a fine-tune of a strata-poly model (the fork: Gemma 4's Kitchoon, Dark-Craving) and your own draft model.

The fingerprint table in front of setup (strata_control/models.plan_import) knows none of the strata-poly models, and a Gemma 4 file's
merges are more than it reads; setup.py's --custom-gguf matches such a file by its architecture.  So --model-path takes the file
through setup's own resolve_custom_gguf, and only when that names a strata-poly family; every other file is checked as before.
--draft-model PATH reaches setup and stays in the preset.  setup's matcher is replaced here by a stand-in: the tiny files below are
not real Gemma 4 files."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest import mock

from tests.terminal.launch_support import HW_12, Home, catalog
from tests.terminal.test_own_files import gguf_bytes

FINE_TUNE = "Dark-Craving-26B-A4B-v3.0.i1-IQ4_XS_q8embd.gguf"


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.base = self.home.base
        self.file = self.base / "RPModel" / FINE_TUNE
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(gguf_bytes([]))             # a header the fingerprint table cannot place, as a real Gemma 4 file
        self.drafter = self.base / "Models" / "gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf"
        self.drafter.parent.mkdir(parents=True)
        self.drafter.write_bytes(b"GGUF")

    def runner(self, **kw):
        from strata_cli.launch import settings as S
        from strata_cli.launch.runner import Runner
        return Runner(S.LaunchSettings(**kw), catalog=catalog(HW_12))

    def matched(self, r, result):
        """setup's resolve_custom_gguf, answering `result` ((family, model) or an exception)."""
        mod = r.catalog.strata.setup_module()
        if isinstance(result, BaseException):
            return mock.patch.object(mod, "resolve_custom_gguf", side_effect=result)
        return mock.patch.object(mod, "resolve_custom_gguf", return_value=(*result, [self.file]))


class FineTune(Base):
    def test_a_fine_tune_of_a_strata_poly_model_is_installed_from_its_file(self):
        r = self.runner(model_path=str(self.file))
        with self.matched(r, ("gemma4", "QAT-UD-Q4_K_XL")):
            r.resolve_own_files()
        self.assertEqual((r.s.family, r.s.model), ("gemma4", "QAT-UD-Q4_K_XL"))
        argv = r.setup_args()
        self.assertEqual(Path(argv[argv.index("--custom-gguf") + 1]), self.file)
        self.assertNotIn("--gguf-dir", argv)
        self.assertEqual(argv[argv.index("--family") + 1], "gemma4")

    def test_its_run_config_is_the_one_setup_names_after_the_file(self):
        r = self.runner(model_path=str(self.file))
        with self.matched(r, ("gemma4", "QAT-UD-Q4_K_XL")):
            r.resolve_own_files()
        self.assertEqual(r.installed_state(), (None, None))
        cfg = r.catalog.root / "strata-gemma4-dark-craving-26b-a4b-v3.0.i1-iq4_xs_q8embd.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text('{"args": ["--model", "%s", "--max-context", "65536"]}' % str(self.file).replace("\\", "\\\\"))
        self.assertEqual(r.installed_state()[0], cfg)

    def test_named_in_a_preset_it_is_still_the_file(self):
        r = self.runner(model_path=str(self.file), family="gemma4", model="qat-ud-q4_k_xl")   # as a preset keeps them
        with self.matched(r, ("gemma4", "QAT-UD-Q4_K_XL")):
            r.resolve_own_files()
        argv = r.setup_args()
        self.assertIn("--custom-gguf", argv)
        self.assertNotIn("--gguf-dir", argv)

    def test_a_file_setup_takes_for_another_engines_model_is_checked_as_before(self):
        from strata_cli.launch.runner import LaunchError
        r = self.runner(model_path=str(self.file))
        with self.matched(r, ("qwen", "Q2_0")), self.assertRaises(LaunchError) as ctx:
            r.resolve_own_files()
        self.assertIn("cannot be used as they are", str(ctx.exception))
        self.assertNotIn("--custom-gguf", r.setup_args())

    def test_a_file_setup_cannot_match_is_refused_as_before(self):
        from strata_cli.launch.runner import LaunchError
        r = self.runner(model_path=str(self.file))
        with self.matched(r, SystemExit(1)), self.assertRaises(LaunchError) as ctx:
            r.resolve_own_files()
        self.assertIn("cannot be used as they are", str(ctx.exception))


class DraftModel(Base):
    def test_your_own_draft_model_reaches_setup(self):
        r = self.runner(model_path=str(self.file), draft_model=str(self.drafter))
        with self.matched(r, ("gemma4", "QAT-UD-Q4_K_XL")):
            r.resolve_own_files()
        argv = r.setup_args()
        self.assertEqual(Path(argv[argv.index("--draft-model") + 1]), self.drafter)

    def test_none_given_none_passed(self):
        r = self.runner(family="gemma4", model="QAT-UD-Q4_K_XL")
        self.assertNotIn("--draft-model", r.setup_args())

    def test_it_is_kept_in_the_preset_and_the_same_as_line(self):
        from strata_cli.launch import settings as S, store
        s = S.LaunchSettings(name="Dark RP", model_path=str(self.file), family="gemma4", model="QAT-UD-Q4_K_XL",
                             draft_model=str(self.drafter), context=131072)
        store.save(s)
        back = store.load("Dark RP")
        self.assertEqual(back.draft_model, str(self.drafter))
        self.assertIn(f'--draft-model "{self.drafter}"', S.flags_text(back))

    def test_the_option(self):
        from strata_cli.launch import cmd
        sp = argparse.ArgumentParser().add_subparsers(dest="cmd")
        parser = cmd.add_parsers(sp)[0]
        a = parser.parse_args(["--model-path", str(self.file), "--draft-model", str(self.drafter)])
        s, _notices = cmd.settings_from_args(a, catalog(HW_12))
        self.assertEqual(Path(s.draft_model), self.drafter)
        a = parser.parse_args(["--model-path", str(self.file)])
        self.assertIsNone(cmd.settings_from_args(a, catalog(HW_12))[0].draft_model)

    def test_the_plan_names_it(self):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import run_cli

        def fine_tune(runner):                          # what poly_fine_tune does for this file (the tests above check that part)
            runner.s.family, runner.s.model, runner.custom_tag = "gemma4", "QAT-UD-Q4_K_XL", "gemma4-dark"
            return True

        with mock.patch.object(Runner, "poly_fine_tune", fine_tune), \
                mock.patch("strata_cli.launch.catalog.Catalog.hardware", return_value=HW_12):
            code, out, err = run_cli(["launch", "--plan", "--model-path", str(self.file), "--draft-model", str(self.drafter),
                                      "--context", "64k", "--vision", "off"])
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"drafting  your own draft model ({self.drafter})", out)
        self.assertIn(f'--draft-model "{self.drafter}"', out)

    def test_its_own_folder_does_not_make_the_install_look_moved(self):
        """The installed config names the draft model outside the fine-tune's folder: that is not a sign the model moved."""
        r = self.runner(model_path=str(self.file), draft_model=str(self.drafter))
        cfg = self.base / "c.json"
        cfg.write_text('{"args": ["--model", "%s", "--draft-model", "%s"]}'
                       % (str(self.file).replace("\\", "\\\\"), str(self.drafter).replace("\\", "\\\\")))
        self.assertTrue(r.installed_serves_request(cfg))


if __name__ == "__main__":
    unittest.main()
