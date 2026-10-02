"""tools/test_fork_mcp.py - the fork's layout in tools/strata_mcp.py (docs/FORK.md): per-OS run configs named
strata-<prefix><tag>-vision|novision.json, their start scripts, the per-OS environment and engine folders, and the
uncensored families' own sizes.  Fork-only, so it never conflicts with upstream's tests.

    python -m unittest tools.test_fork_mcp -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import setup as S  # noqa: E402
import strata_mcp as M  # noqa: E402

HW = {"os": "TestOS 1", "ram_gb": 64.0, "cpu": {"name": "Fake CPU", "cores": 12, "avx2": True, "avx512": False},
      "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "arch": "86", "vendor": "nvidia",
                "problem": None, "usable": True}]}
EXT = "bat" if M.WIN else "sh"
OTHER = "linux-" if M.WIN else ""                     # the other system's configs in a dual-boot folder


class ForkLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "Strata"
        self.root.mkdir()
        self.env = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
        os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"] = str(Path(self.tmp.name) / "config")
        self.s = M.Strata(self.root)
        self.s._setup, self.s._setup_tried = S, True    # the real tables: the fork's families and sizes
        self.s._hardware = lambda: dict(HW)
        self.s.disk_free = lambda p: 1000.0
        self.tools = M.Tools(self.s)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def config(self, name, mtime):
        p = self.root / name
        p.write_text(json.dumps({"exe": sys.executable, "args": ["--max-context", "131072"], "port": 8080,
                                 "model_name": "orca-uncensored-iq2_m"}))
        os.utime(p, (mtime, mtime))
        return p

    def install_orca(self):
        pre = M.CFG_PREFIX
        novision = self.config(f"strata-{pre}orca-iq2_m-novision.json", 1000)
        vision = self.config(f"strata-{pre}orca-iq2_m-vision.json", 2000)
        self.config(f"strata-{pre}orca-iq2_m-vision.shared-settings.json", 3000)   # the chat settings, not a model
        self.config(f"strata-{OTHER}orca-iq2_m-vision.json", 4000)                 # the other OS's
        (self.root / f"run-orca-iq2_m-vision.{EXT}").write_text("")
        return vision, novision

    def test_configs_are_this_systems_models_only(self):
        vision, novision = self.install_orca()
        self.assertEqual(self.s.configs(), [vision, novision])

    def test_a_config_names_its_start_script(self):
        vision, novision = self.install_orca()
        d = self.s.describe_config(vision)
        self.assertEqual(d["model"], "orca-iq2_m-vision")
        self.assertEqual(d["start_script"], f"run-orca-iq2_m-vision.{EXT}")
        self.assertIsNone(self.s.describe_config(novision)["start_script"])     # none written for it here

    def test_a_model_is_found_with_or_without_its_images_setting(self):
        vision, novision = self.install_orca()
        self.assertEqual(self.s.find_config("orca-iq2_m"), vision)            # the most recently used
        self.assertEqual(self.s.find_config("orca-iq2_m-novision"), novision)
        self.assertEqual(self.s.find_config(None), vision)

    def test_environment_and_engine_folders(self):
        self.assertEqual(self.s.venv_python().parent.parent.name, ".venv" if M.WIN else ".venv-linux")
        eng = self.root / ("engine" if M.WIN else "engine-linux")
        eng.mkdir()
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.34", "source": "prebuilt"}))
        self.assertEqual(self.s.engine_info()["version"], "0.1.34")

    def test_the_uncensored_families_use_their_own_sizes(self):
        models, families, _c, _src = self.s.tables()
        self.assertEqual(self.s.sizes_of(models, "orca"), list(S.FAMILIES["orca"]["sizes"]))
        self.assertEqual(self.s.sizes_of(models, "qwen"),
                         [m for m, d in S.MODELS.items() if "qwen" in d.get("families", ("qwen", "swift"))])

    def test_install_plan_for_a_gated_uncensored_model(self):
        self.install_orca()
        plan, args = self.tools.install_plan("orca", "IQ2_M", None, None, None, None, None, None, None, None, None)
        self.assertEqual(plan["download_gb"], S.FAMILIES["orca"]["sizes"]["IQ2_M"]["download_gb"])
        self.assertTrue(plan["already_installed"])
        self.assertIn("HF_TOKEN", plan["gated"])
        self.assertEqual(args[args.index("--family") + 1], "orca")

    def test_model_table_lists_the_uncensored_sizes(self):
        self.install_orca()
        orca = next(f for f in self.s.model_table(HW) if f["family"] == "orca")
        iq2m = next(x for x in orca["sizes"] if x["model"] == "IQ2_M")
        self.assertTrue(iq2m["installed"])
        self.assertEqual(iq2m["id"], "orca-iq2_m")
        self.assertNotIn("low-RAM", iq2m["on_this_pc"] or "")              # setup has no low-RAM mode for them


if __name__ == "__main__":
    unittest.main()
