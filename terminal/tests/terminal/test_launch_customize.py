"""`strata launch --customize`: the screens open at the Context step with the command line's choices, the model's own steps skipped."""

from __future__ import annotations

import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.app import LaunchApp
from tests.terminal.test_launch_tui import Base, flat, make_ctx, press


class StartAtTests(Base):
    def settings(self):
        return S.LaunchSettings(family="qwen", model="IQ3_XXS", context=65536)

    def open(self, coro, **kw):
        """`run_app` of the base class takes only auto_run: build the app here so start_at reaches it."""
        import asyncio
        app = LaunchApp(make_ctx(settings=self.settings()), **kw)

        async def go():
            async with app.run_test(size=(110, 60)) as pilot:
                await pilot.pause(0.2)
                return await coro(app, pilot)

        return asyncio.run(go())

    def test_start_at_context_skips_the_model_and_folder_steps(self):
        async def case(app, pilot):
            self.assertEqual(app.current, "context")
            t = flat(app)
            self.assertRegex(t, r"Step \d+ of \d+: Context")
            self.assertNotIn("Which model?", t)
            self.assertEqual(app.ctx.settings.model, "IQ3_XXS")      # the command line's choice is kept
            self.assertEqual(app.ctx.settings.context, 65536)

        self.open(case, start_at="context")

    def test_back_goes_to_the_skipped_steps_and_next_goes_on(self):
        async def case(app, pilot):
            await press(pilot, "escape")
            self.assertEqual(app.current, "folder")
            await press(pilot, "enter")
            self.assertEqual(app.current, "context")
            await press(pilot, "enter")
            self.assertEqual(app.current, "pictures")

        self.open(case, start_at="context")

    def test_an_unknown_step_opens_the_first_one(self):
        async def case(app, pilot):
            self.assertEqual(app.current, "model")

        self.open(case, start_at="nonsense")

    def test_without_start_at_the_start_screen_comes_first(self):
        async def case(app, pilot):
            self.assertEqual(app.current, "start")

        self.open(case)


class CommandLineTests(unittest.TestCase):
    def parser(self):
        import argparse
        from strata_cli.launch import cmd
        top = argparse.ArgumentParser(prog="strata")
        cmd.add_parsers(top.add_subparsers(dest="command"))
        return top

    def test_the_flag_is_parsed_and_off_by_default(self):
        top = self.parser()
        self.assertTrue(top.parse_args(["launch", "--customize", "--model-path", "m.gguf"]).customize)
        self.assertFalse(top.parse_args(["launch", "--model-path", "m.gguf"]).customize)

    def test_the_help_says_where_it_starts(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            self.parser().parse_args(["launch", "--help"])
        text = " ".join(out.getvalue().split())
        self.assertIn("--customize", text)
        self.assertIn("Context step", text)


if __name__ == "__main__":
    unittest.main()
