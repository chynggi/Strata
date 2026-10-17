"""Fork-only tests for setup.py's uncensored families (no GPU, no network, nothing installed): the shard names the
fork's community GGUFs are published under - they carry the shard count in the name ({n:05d}), which upstream's
`model_file` did not pass after the v0.1.40.1 merge.

    python -m unittest tools.test_fork_setup
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class UncensoredShardNames(unittest.TestCase):
    """The published names (verified against the Hub's file lists on 2026-10): -{i:05d}-of-{n:05d}."""

    def test_orcarouter_iq2_m(self):
        fam = setup.FAMILIES["orca"]
        names = [setup.model_file(fam, "IQ2_M", i) for i in range(1, setup.model_shards(fam, "IQ2_M") + 1)]
        self.assertEqual(names, ["Qwen3.8-Flash-Next-Uncensored-IQ2_M-00001-of-00002.gguf",
                                 "Qwen3.8-Flash-Next-Uncensored-IQ2_M-00002-of-00002.gguf"])

    def test_rvn_iq3_xs(self):
        fam = setup.FAMILIES["rvn"]
        names = [setup.model_file(fam, "IQ3_XS", i) for i in range(1, setup.model_shards(fam, "IQ3_XS") + 1)]
        self.assertEqual(len(names), 8)
        self.assertEqual(names[0], "RVN-Qwen3.8-Flash-Next-IQ3_XS-00001-of-00008.gguf")
        self.assertEqual(names[-1], "RVN-Qwen3.8-Flash-Next-IQ3_XS-00008-of-00008.gguf")

    def test_every_choice_formats_its_shard_names(self):
        """Each family's pattern makes `model_shards` distinct names whose own i-of-N suffix matches them."""
        for f, fam in setup.FAMILIES.items():
            if "file" not in fam:                              # upstream's ready-made packs (nvfp4) list their files
                continue
            sizes = fam.get("sizes") or {m: d for m, d in setup.MODELS.items()
                                         if f in d.get("families", ("qwen", "swift"))}
            for model in sizes:
                n = setup.model_shards(fam, model)
                names = [setup.model_file(fam, model, i) for i in range(1, n + 1)]
                self.assertEqual(len(set(names)), n, f"{f} {model}: {names}")
                for i, name in enumerate(names, 1):
                    m = setup.SHARD_NAME.search(name)
                    if m:                                      # the single-file families (mrad) have no suffix
                        self.assertEqual(int(m.group(1)), i, name)
                        self.assertEqual(int(m.group(2)), n, name)


if __name__ == "__main__":
    unittest.main()
