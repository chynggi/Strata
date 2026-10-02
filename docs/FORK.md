# This fork vs upstream

This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). It tracks upstream
`main` and adds a few things on top. This file lists what the fork changes, so a later upstream merge can
resolve conflicts the same way every time, and so nothing upstream adds is dropped by accident.

## What the fork adds

| Area | Files | What it does |
| --- | --- | --- |
| Uncensored models | `setup.py` (`FAMILIES`: `orca`, `mrad`, `rvn`), `tools/iq_pack.py` | One-click install of community "abliterated" GGUFs. `orca` is gated (a Hugging Face token is asked for); each family carries its own `sizes` dict. |
| Community-GGUF packing | `tools/iq_pack.py` | `--compat-bf16` (the families' `compat_bf16`) makes a quantized PLE key BF16 in the pack, which the engine keeps (#326; verified on OrcaRouter IQ2_M, whose IQ2_S key once needed the shard rewritten by the removed `tools/ple_key_bf16.py`); quantized tensors the engine cannot serve natively (e.g. Q2_K) are dequantized to BF16 (`served_natively` / `dequant_bf16`). Layers split across shards and atomic pack writes are upstream's since 0.1.34 (`native_experts.txt` v4). |
| Dual-boot layout | `setup.py` (`engine_dir()`, `build_dir()`, `vision_build_dir()`, `CFG_PREFIX`), `setup.sh`, `.gitignore` | Windows and Linux share one folder: each OS keeps its own engine (`engine/` vs `engine-linux/`), CMake cache (`build*/` vs `build-linux*/`), Python env (`.venv` vs `.venv-linux/`), run config and log (`strata-*.json` vs `strata-linux-*.json`). Model data stays shared. |
| MCP server | `tools/strata_mcp.py`, `tools/test_fork_mcp.py` | The dual-boot layout (`VENV`, `ENGINE`, `CFG_PREFIX`): this OS's run configs only (not `*.shared-settings.json`), model ids `<tag>-vision|novision` with their `run-<id>` scripts (`<tag>` alone finds the latest), and the uncensored families' own `sizes` (`size_table`); a gated family's install plan says it needs `HF_TOKEN`. |
| Line endings | `.gitattributes`, `.gitignore` | `* text=auto eol=lf` so a Windows editor cannot turn the tree into CRLF (an earlier commit did, and every later merge conflicted on every line). |

Fork-only files (upstream has none of these, so they never conflict): `docs/FORK.md`,
`scripts/merge-upstream.sh`, `scripts/merge-upstream.ps1`, `tools/test_fork_mcp.py`,
`MoreSimpleStart.*`.

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
   (the fork's families, `engine_dir()`/`CFG_PREFIX`, and its `iq_pack.py` helpers are the usual places).
2. `git add` the resolved files and `git commit` - rerere records each resolution for next time.
3. Run the Python tests: `python -m unittest discover -s tools -p "test_*.py"`.
4. Do not push unresolved forks of upstream's own files without checking this file.

### Resolving the usual conflicts

- **`setup.py` / `FAMILIES`**: keep upstream's new families and the fork's `orca` / `mrad` / `rvn`; take
  upstream's new fields (multi-GPU, calibration) and keep `engine_dir()`, `build_dir()`, `CFG_PREFIX`.
  The fork's families' Hugging Face revisions are pinned in the `HF_REVISIONS.update` block below upstream's.
- **`setup.py` / engine paths**: upstream writes `ROOT / "engine"` and `ROOT / "build"`; those are the
  fork's `engine_dir()` and `build_dir()` - also in code that merged without a conflict (grep for them).
  They are functions so they follow `ROOT`: the tests point `ROOT` at a temporary folder, and a constant
  kept the real `engine/`, which the tests' fake engines overwrote.
- **`setup.py` / model choice**: upstream picks sizes from `MODELS`; the fork uses `fam.get("sizes") or
  {... MODELS ...}` and passes `--compat-bf16` for families that set `compat_bf16`.
- **`setup.py` / config names**: `strata-{CFG_PREFIX}{tag}-vision|novision` for the config and log,
  `run-{tag}-vision|novision` for the start script (one of each per images setting).
- **`setup.py` / `MODELS[model]` in `main()`**: upstream's new code reads `MODELS[model]`; in `main()` it is the
  fork's `sizes[model]` (helpers that read it take a `sizes=MODELS` argument: `confirm_paging`, `ctx_ram_need`,
  `ram_ctx`) (the fork's families have sizes MODELS does not, e.g. `IQ2_M`). The low-RAM helpers
  (`low_ram_*`) stay on `MODELS`, and the low-RAM mode is off for families that carry their own `sizes`.
- **`setup.py` / HIP engine**: upstream's `build_engine_hip`, `get_prebuilt_hip` and `ensure_engine_for` write
  `ROOT / "engine"`; that is `engine_dir()` (`engine/` on Windows, `engine-linux/` on Linux).
- **`setup.py` / `download`**: the fork's `token=` is passed only for a gated model, so upstream's tests' fake
  `download(url, dst, what=None)` keeps working. `write_run_script` keeps upstream's three arguments (the
  vision suffix comes from the config's name).
- **`setup.sh`**: the environment is `.venv-linux`, not `.venv`.
- **`tools/iq_pack.py`**: take upstream's file (FORM conversions, `native_experts.txt` v4, the experts.bin
  sidecar), then keep the fork's `served_natively` / `dequant_bf16` and the
  BF16 fallback in `index_standalone`'s loop for a quantized tensor with no FORM entry that the engine cannot
  serve; `NATIVE_TYPES` follows `native_mmvq_supported` in `src/kernels/cuda/native_mmvq.cu`.
- **Tests**: `tools/test_setup_golden.json` has the fork's `-vision|novision` log names;
  `tools/test_setup_unsloth.py` reads `strata-unsloth-ud-q4_k_xl-novision.json`; `tools/strata_mcp.py`'s
  `FALLBACK_FAMILIES` lists the fork's families.
- **`docs/DETAILS.md`, `README.md`**: upstream's `Strata-data` text stays; the fork's dual-boot section and
  uncensored table stay.