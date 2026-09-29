# This fork vs upstream

This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). It tracks upstream
`main` and adds a few things on top. This file lists what the fork changes, so a later upstream merge can
resolve conflicts the same way every time, and so nothing upstream adds is dropped by accident.

## What the fork adds

| Area | Files | What it does |
| --- | --- | --- |
| Uncensored models | `setup.py` (`FAMILIES`: `orca`, `mrad`, `rvn`), `tools/ple_key_bf16.py`, `tools/iq_pack.py` | One-click install of community "abliterated" GGUFs. `orca` is gated (a Hugging Face token is asked for); each family carries its own `sizes` dict. |
| Community-GGUF packing | `tools/iq_pack.py` | A PLE key quantized to anything but Q2_0 is stored as BF16; quantized tensors the engine cannot serve natively are dequantized to BF16; a layer whose experts straddle two shards is written to `experts.bin`. Packs are written under `.part` names and renamed last. |
| Dual-boot layout | `setup.py` (`ENGINE_DIR`, `BUILD_DIR`, `VISION_BUILD_DIR`, `CFG_PREFIX`), `setup.sh`, `.gitignore` | Windows and Linux share one folder: each OS keeps its own engine (`engine/` vs `engine-linux/`), CMake cache (`build*/` vs `build-linux*/`), Python env (`.venv` vs `.venv-linux/`), run config and log (`strata-*.json` vs `strata-linux-*.json`). Model data stays shared. |
| Line endings | `.gitattributes`, `.gitignore` | `* text=auto eol=lf` so a Windows editor cannot turn the tree into CRLF (an earlier commit did, and every later merge conflicted on every line). |

Fork-only files (upstream has none of these, so they never conflict): `docs/FORK.md`,
`scripts/merge-upstream.sh`, `scripts/merge-upstream.ps1`, `tools/ple_key_bf16.py`.

## Keeping merges small

The fork touches only the files above. Everything else is upstream's; do not edit it locally without a
reason, and keep changes inside existing fork blocks when one exists.

`git rerere` is the main tool. It records how each conflict hunk was resolved and replays that resolution
the next time the same hunk conflicts, so the work of resolving a merge is done once:

```sh
git config rerere.enabled true
git config rerere.autoupdate true      # stage the replayed resolutions automatically
git config merge.conflictStyle zdiff3  # show the common ancestor in conflicts (easier, better rerere)
```

These are local settings. `scripts/merge-upstream.sh` (Linux/macOS) and `scripts/merge-upstream.ps1`
(Windows) set them and do the fetch + merge.

## Merging upstream

```sh
scripts/merge-upstream.sh          # or: scripts/merge-upstream.ps1
```

Then, if there are conflicts:

1. Resolve them, keeping both sides where upstream only added code and the fork only renamed or added
   (the fork's families, `ENGINE_DIR`/`CFG_PREFIX`, and its `iq_pack.py` helpers are the usual places).
2. `git add` the resolved files and `git commit` - rerere records each resolution for next time.
3. Run the Python tests: `python -m unittest discover -s tools -p "test_*.py"`.
4. Do not push unresolved forks of upstream's own files without checking this file.

### Resolving the usual conflicts

- **`setup.py` / `FAMILIES`**: keep upstream's new families and the fork's `orca` / `mrad` / `rvn`; take
  upstream's new fields (multi-GPU, calibration) and keep `ENGINE_DIR`, `BUILD_DIR`, `CFG_PREFIX`.
- **`setup.py` / engine paths**: upstream writes `ROOT / "engine"` and `ROOT / "build"`; those are the
  fork's `ENGINE_DIR` and `BUILD_DIR`.
- **`setup.py` / model choice**: upstream picks sizes from `MODELS`; the fork uses `fam.get("sizes") or
  {... MODELS ...}` and passes `--compat-bf16` for families that set `compat_bf16`.
- **`setup.py` / config names**: `strata-{CFG_PREFIX}{tag}` for the config and log.
- **`setup.sh`**: the environment is `.venv-linux`, not `.venv`.
- **`tools/iq_pack.py`**: keep upstream's `n_expert` and `--compat-bf16`; keep the fork's
  `served_natively` / `dequant_bf16` fallback in the per-tensor loop.
- **`docs/DETAILS.md`, `README.md`**: upstream's `Strata-data` text stays; the fork's dual-boot section and
  uncensored table stay.