# This fork vs upstream

This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). It tracks upstream
`main` and adds a few things on top. This file lists what the fork changes, so a later upstream merge can
resolve conflicts the same way every time, and so nothing upstream adds is dropped by accident.

## What the fork adds

| Area | Files | What it does |
| --- | --- | --- |
| Uncensored models | `setup.py` (`FAMILIES`: `orca`, `mrad`, `rvn`), `tools/iq_pack.py` | One-click install of community "abliterated" GGUFs. `orca` is gated (a Hugging Face token is asked for); each family carries its own `sizes` dict. |
| Community-GGUF packing | `tools/iq_pack.py` | `--compat-bf16` (the families' `compat_bf16`) makes a quantized PLE key BF16 in the pack, which the engine keeps (#326; verified on OrcaRouter IQ2_M, whose IQ2_S key once needed the shard rewritten by the removed `tools/ple_key_bf16.py`); quantized tensors the engine cannot serve natively (e.g. Q2_K) are dequantized to BF16 (`served_natively` / `dequant_bf16`). Layers split across shards and atomic pack writes are upstream's since 0.1.34 (`native_experts.txt` v4). |
| Dual-boot layout | `setup.py` (`engine_dir(toolkit)`, `build_dir(toolkit)`, `vision_build_dir(toolkit)`, `CFG_PREFIX`), `setup.sh`, `.gitignore` | Windows and Linux share one folder: each OS keeps its own engine (`engine/` vs `engine-linux/`, and upstream's `engine-cuda12/` vs `engine-cuda12-linux/`), CMake cache (`build*/` vs `build-linux*/`), Python env (`.venv` vs `.venv-linux/`), run config and log (`strata-*.json` vs `strata-linux-*.json`). Model data stays shared. |
| MCP server | `tools/strata_mcp.py`, `tools/test_fork_mcp.py` | The dual-boot layout (`VENV`, `ENGINE`, `CFG_PREFIX`): this OS's run configs only (not `*.shared-settings.json`), model ids `<tag>-vision|novision` with their `run-<id>` scripts (`<tag>` alone finds the latest), and the uncensored families' own `sizes` (`size_table`; also where upstream's `install_plan` reads `vision`/`experimental`, which it writes as `models[model]`); a gated family's install plan says it needs `HF_TOKEN`. |
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
  kept the real `engine/`, which the tests' fake engines overwrote.  Since v0.1.40 each takes a `toolkit`
  (upstream's experimental CUDA 12 engine lives in `engine-cuda12/`) and the OS suffix goes on top of it
  (`engine-cuda12-linux/`); keep **one** definition - two had merged into the file and Python silently kept
  upstream's, which dropped the dual-boot naming.  `config_toolkit` compares against `engine_dir(12).name`.
- **`setup.py` / model choice**: upstream picks sizes from `MODELS`; the fork uses `fam.get("sizes") or
  {... MODELS ...}` and passes `--compat-bf16` for families that set `compat_bf16`. `names = list(sizes)`; the
  "no such size" check is upstream's #444 message (`has no {model} model file`), with `MODELS.get(model, {})`
  for its "or <family>" hint so a fork-family size (`IQ2_M`, `IQ3_XS`) cannot raise `KeyError`. The budget
  branch is upstream's `budget, q4_split = None, False`, guarded by the fork's `sizes[model].get("budget")`.
- **`setup.py` / config list**: upstream's `model_config_files` (a folder's `strata-*.json` minus the
  `*.shared-settings.json` chat settings the server keeps) is what everything calls; the fork's `run_configs`
  was the same function and was dropped in the v0.1.40.1 merge, so do not bring it back. `installed_configs` is
  the fork's per-OS dual-boot filter over it (`[p for p in model_config_files(ROOT) if model_config(p)]`, then
  the `strata-linux-` test), and the fork's `config_label` is kept.
- **`setup.py` / config names**: `strata-{CFG_PREFIX}{tag}-vision|novision` for the config and log,
  `run-{tag}-vision|novision` for the start script (one of each per images setting).
- **`setup.py` / `MODELS[model]` in `main()`**: upstream's new code reads `MODELS[model]`; in `main()` it is the
  fork's `sizes[model]` (helpers that read it take a `sizes=MODELS` argument: `confirm_paging`, `ctx_ram_need`,
  `ram_ctx`) (the fork's families have sizes MODELS does not, e.g. `IQ2_M`). The low-RAM helpers
  (`low_ram_*`) stay on `MODELS`, and the low-RAM mode is off for families that carry their own `sizes`.
- **`setup.py` / HIP engine**: upstream's `build_engine_hip`, `get_prebuilt_hip` and `ensure_engine_for` write
  `ROOT / "engine"`; that is `engine_dir()` (`engine/` on Windows, `engine-linux/` on Linux; the HIP engine is
  never the CUDA 12 one, so it stays on the default toolkit 13).
- **`setup.py` / `download`**: the fork's `token=` is passed only for a gated model, so upstream's tests' fake
  `download(url, dst, what=None)` keeps working. `write_run_script` keeps upstream's three arguments (the
  vision suffix comes from the config's name).
- **`setup.sh`**: the environment is `.venv-linux`, not `.venv`.
- **`tools/iq_pack.py`**: take upstream's file (FORM conversions, `native_experts.txt` v4, the experts.bin
  sidecar), then keep the fork's `served_natively` / `dequant_bf16` and the
  BF16 fallback in `index_standalone`'s loop for a quantized tensor with no FORM entry that the engine cannot
  serve; `NATIVE_TYPES` follows `native_mmvq_supported` in `src/kernels/cuda/native_mmvq.cu`.
- **Tests**: `tools/test_setup_golden.json` has the fork's `-vision|novision` log names (every golden case
  upstream adds needs one too); `tools/test_setup_unsloth.py` reads `strata-unsloth-ud-q4_k_xl-novision.json`
  and takes the Unsloth family's menu number from `setup.FAMILIES` (the fork's families come first, so a
  hard-coded `4)` is wrong); `tools/test_setup_config.py` (upstream's, new in v0.1.40) writes and expects
  `strata-q2_0-novision.json`, or `-vision.json` when the case uses `--vision`; `tools/strata_mcp.py`'s
  `FALLBACK_FAMILIES` lists the fork's families.
- **`docs/DETAILS.md`, `README.md`**: upstream's `Strata-data` text stays; the fork's dual-boot section and
  uncensored table stay.

## Merge log

Where each upstream `main` was folded into the fork (the fork's own merges in `git log --merges main`):

| Upstream | Fork merge commit | What it brought / the `setup.py` conflicts resolved |
| --- | --- | --- |
| engine 0.1.3-0.1.6 | `e7db7b1` | IQ3_S, KV streaming, conversation cache |
| v0.1.22 | `c2f423b` | |
| v0.1.30 | `469ee40` | |
| v0.1.34 | `f68c1c6` | `tools/iq_pack.py` (FORM conversions, `native_experts.txt` v4) and `setup.py` (budget mode, `gguf_dir_shards`, the engine/build dir functions) |
| v0.1.38 | `69aa0e4` | engine 0.1.35-0.1.38; `setup.py` only (the config helpers, the #444 model check, `budget, q4_split`), see the bullets above |
| v0.1.40.1 | `9558fea` | engine 0.1.39-0.1.40.1: the experimental CUDA 12 engine, `model_file`/`model_shards` (#621), `model_config_files` (#346), the `low_ram_wanted` / KV-streaming refactors (#608 #642), the engine rollback (#670), `SECURITY.md`, seven translated READMEs. Upstream had force-pushed a rewritten history, so the base was re-anchored first - see below. `setup.py`: one `engine_dir(toolkit)`/`build_dir(toolkit)`/`vision_build_dir(toolkit)` for both the CUDA 12 engine and the dual-boot folders, `sizes[model]` for upstream's new `MODELS[model]` in `main()`, `model_config_files` for the dropped `run_configs`, the fork's config names for upstream's new `old_cfg`; `tools/strata_mcp.py`'s `install_plan` reads the family's size dict |

`git rerere` replayed the recorded `ROOT / "engine"` -> `engine_dir()` and `ROOT / "build"` -> `build_dir()`
renames onto upstream's new code in the v0.1.38 merge, so those hunks merged without a conflict. The Python
tests pass with the fork's `.venv` (467 tests since v0.1.40.1; 294 before it).

### The v0.1.40.1 merge: upstream rewrote its history

Upstream force-pushed a history whose commits are the fork's copies with the Claude attribution trailers
stripped from every message. The trees and the dates are identical, the SHAs are not, so the two histories had
**no commit in common** and `git merge upstream/main` refused to run at all ("unrelated histories"). Nothing
was lost: the fork's v0.1.38 commit (`99f3dbd`) and upstream's (`09817da`) are the same tree, so re-anchoring
one to the other turned the merge back into an ordinary one:

```sh
git replace --graft 09817da 99f3dbd    # upstream's v0.1.38 -> the fork's copy of it (same tree)
scripts/merge-upstream.sh              # or .ps1: a normal merge, base v0.1.38, conflicts per this file
git replace -d 09817da                 # AFTER committing: the merge commit is what joins the histories
```

Drop the replace ref once the merge is committed: kept, it would make the *next* merge use v0.1.38 as its base
again and re-apply every release since. With it gone, `git merge-base main upstream/main` is upstream's tip
(the merge commit's second parent), so later merges are ordinary. If upstream rewrites again, find the
upstream commit matching a fork commit by the tree and the dates, with the message differing only in the
trailers (`git log --format='%T %s' upstream/main` against the fork's).