"""The fork's uncensored families (orca / mrad / rvn, setup.py FAMILIES "sizes") get the same fit rule as the others: the download minus the PLE lookup table.

Their GGUFs carry the table inside the model file (per_layer_token_embd.weight, IQ4_NL, 28,800,138,240 bytes in each of orca IQ2_M / IQ3_XXS, mrad IQ3_S and
rvn IQ3_XS, read from their GGUF headers; the same tensor as the GSQ-RCO table shard), so fit_label did not know them and counted the whole download as "Needs (RAM + VRAM)": a 64 GB PC with a 12 GB card read "may run slower".
"""
from __future__ import annotations

import unittest

from strata_control import fit_label


class ForkFamilies(unittest.TestCase):
    def test_the_table_is_taken_off_the_download(self):
        import setup
        want = {("orca", "IQ2_M"): 51.3, ("orca", "IQ3_XXS"): 56.4, ("mrad", "IQ3_S"): 60.1, ("rvn", "IQ3_XS"): 57.2}
        for (family, model), gb in want.items():
            d = setup.FAMILIES[family]["sizes"][model]
            self.assertEqual(fit_label.files_gb(family, model, d), gb, (family, model))

    def test_a_64_gb_pc_with_a_12_gb_card(self):
        import setup
        for family in ("orca", "mrad", "rvn"):
            for model, d in setup.FAMILIES[family]["sizes"].items():
                self.assertEqual(fit_label.label(fit_label.files_gb(family, model, d), 62.0, 12.0), fit_label.FITS, (family, model))


if __name__ == "__main__":
    unittest.main()
