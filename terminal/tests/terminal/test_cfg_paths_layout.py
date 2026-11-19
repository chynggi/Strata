"""A run config or launch preset outside a source clone that names a file at the clone's root of the layout before 0.2.16
(<clone>/data/experimental-speed-projection/...gguf:1.0) is pointed at <clone>/terminal/... once the file has moved there.
Before, the engine of a saved Gyro-M preset stopped at "control vector: cannot open C:\\Tools\\Strata\\data\\...".
Plain unittest with tempfile (pytest's tmp_path base folder cannot be read on the owner PC: WinError 5, 11-10)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import cfg_paths  # noqa: E402

VEC = "experimental-speed-projection"
GGUF = "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"


class LayoutMove(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.clone = Path(tmp.name) / "Strata"
        (self.clone / "engine").mkdir(parents=True)
        (self.clone / "engine" / "CMakeLists.txt").write_text("")
        (self.clone / "terminal" / "data" / VEC).mkdir(parents=True)
        (self.clone / "terminal" / "setup.py").write_text("")
        (self.clone / "terminal" / "data" / VEC / GGUF).write_bytes(b"GGUF")
        (self.clone / "terminal" / "data" / "expert-profile.bin").write_bytes(b"STRP")
        self.old = str(self.clone / "data" / VEC / GGUF)
        self.new = str(self.clone / "terminal" / "data" / VEC / GGUF)

    def test_a_moved_file_is_found_in_terminal_and_its_scale_is_kept(self):
        self.assertEqual(cfg_paths.resolve_layout_path(self.old), self.new)
        self.assertEqual(cfg_paths.resolve_layout_path(self.old + ":1.0"), self.new + ":1.0")
        self.assertEqual(cfg_paths.resolve_layout_path(self.old + ":-0.5"), self.new + ":-0.5")
        self.assertEqual(cfg_paths.resolve_value(self.old + ":1.0"), self.new + ":1.0")

    def test_nothing_else_is_touched(self):
        self.assertIsNone(cfg_paths.resolve_layout_path(self.new), "an existing file stays")
        self.assertIsNone(cfg_paths.resolve_layout_path(str(self.clone / "data" / "not-there.bin")), "no counterpart: not guessed")
        self.assertIsNone(cfg_paths.resolve_layout_path(str(self.clone / "models" / "a.gguf")), "not a Terminal entry")
        self.assertIsNone(cfg_paths.resolve_layout_path("data/expert-profile.bin"), "a relative path stays")
        (self.clone / "engine" / "CMakeLists.txt").unlink()
        self.assertIsNone(cfg_paths.resolve_layout_path(self.old), "not a 0.2.16 clone")
        for value in ("auto", "int8", "4", "per-layer", 32768):
            self.assertEqual(cfg_paths.resolve_value(value), value)

    def test_a_run_config_is_resolved_and_repaired_once(self):
        cfg = {"cwd": str(self.clone / "terminal"), "log": str(self.clone / "data" / "x.log"),
               "args": ["--expert-profile", str(self.clone / "data" / "expert-profile.bin"),
                        "--control-vector-scaled", self.old + ":1.0", "--cvec-mode", "project"],
               "user_args": ["--control-vector-scaled", self.old + ":1.0"]}
        new, found, lost = cfg_paths.resolve_config(cfg)
        self.assertEqual(new["args"][1], str(self.clone / "terminal" / "data" / "expert-profile.bin"))
        self.assertEqual(new["args"][3], self.new + ":1.0")
        self.assertEqual(new["user_args"][1], self.new + ":1.0")
        self.assertEqual(new["log"], cfg["log"], "a path the engine writes is left alone")
        self.assertEqual(len(found), 3)
        self.assertEqual(lost, [])
        path = Path(self.clone.parent) / "launch.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        notes, problems = cfg_paths.repair([path])
        self.assertEqual(len(notes), 1)
        self.assertEqual(problems, [])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["user_args"][1], self.new + ":1.0")
        self.assertEqual(cfg_paths.repair([path]), ([], []), "nothing left to change")

    def test_a_launch_preset_extra_is_resolved_on_load(self):
        from strata_cli.launch import store
        presets = Path(self.clone.parent) / "presets"
        presets.mkdir()
        (presets / "gyro.json").write_text(json.dumps({"name": "gyro", "family": "gyro", "model": "Gyro-M",
                                                       "extra": {"--control-vector-scaled": self.old + ":1.0", "--cvec-mode": "project"}}),
                                           encoding="utf-8")
        before = os.environ.get("STRATA_LAUNCH_PRESET_DIR")
        os.environ["STRATA_LAUNCH_PRESET_DIR"] = str(presets)
        try:
            s = store.load("gyro")
        finally:
            if before is None:
                os.environ.pop("STRATA_LAUNCH_PRESET_DIR", None)
            else:
                os.environ["STRATA_LAUNCH_PRESET_DIR"] = before
        self.assertEqual(s.extra["--control-vector-scaled"], self.new + ":1.0")
        self.assertEqual(s.extra["--cvec-mode"], "project")


if __name__ == "__main__":
    unittest.main()
