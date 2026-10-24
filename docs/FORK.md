# This fork vs upstream

This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). It tracks upstream
`main` and adds a few things on top. This file lists what the fork changes, so a later upstream merge can
resolve conflicts the same way every time, and so nothing upstream adds is dropped by accident.

## What the fork adds

| Area | Files | What it does |
| --- | --- | --- |
| Uncensored models | `setup.py` (`FAMILIES`: `orca`, `mrad`, `rvn`), `tools/iq_pack.py` | One-click install of community "abliterated" GGUFs. `orca` is gated (a Hugging Face token is asked for); each family carries its own `sizes` dict. |
| Community-GGUF packing | `tools/iq_pack.py` | `--compat-bf16` (the families' `compat_bf16`) makes a quantized PLE key BF16 in the pack, which the engine keeps (#326; verified on OrcaRouter IQ2_M, whose IQ2_S key once needed the shard rewritten by the removed `tools/ple_key_bf16.py`); quantized tensors the engine cannot serve natively (e.g. Q2_K) are dequantized to BF16 (`served_natively` / `dequant_bf16`). Layers split across shards and atomic pack writes are upstream's since 0.1.34 (`native_experts.txt` v4). |
| Dual-boot layout | `setup.py` (`engine_dir(toolkit)`, `build_dir(toolkit)`, `vision_build_dir(toolkit)`, `CFG_PREFIX`), `setup.sh`, `docker-entrypoint.sh`, `.gitignore` | Windows and Linux share one folder: each OS keeps its own engine (`engine/` vs `engine-linux/`, and upstream's `engine-cuda12/` vs `engine-cuda12-linux/`), CMake cache (`build*/` vs `build-linux*/`), Python env (`.venv` vs `.venv-linux/`), run config and log (`strata-*.json` vs `strata-linux-*.json`). Model data stays shared. |
| MCP server | `tools/strata_mcp.py`, `tools/test_fork_mcp.py` | The dual-boot layout (`VENV`, `ENGINE`, `CFG_PREFIX`): this OS's run configs only (not `*.shared-settings.json`), model ids `<tag>-vision|novision` with their `run-<id>` scripts (`<tag>` alone finds the latest), and the uncensored families' own `sizes` (`size_table`; also where upstream's `install_plan` reads `vision`/`experimental`, which it writes as `models[model]`); a gated family's install plan says it needs `HF_TOKEN`. |
| Line endings | `.gitattributes`, `.gitignore` | `* text=auto eol=lf` so a Windows editor cannot turn the tree into CRLF (an earlier commit did, and every later merge conflicted on every line). |
| Windows locale | `tools/strata_bench.py` | `project_version()` reads the BOM'd `CMakeLists.txt` as UTF-8 (`encoding="utf-8-sig"`, as `setup.py`'s `source_version()` already does). Upstream reads it through the locale encoding, which is cp949 on a Korean Windows, so `BENCH.bat` stopped with `UnicodeDecodeError` before measuring anything (new in 0.1.43). Report it upstream and drop the change when it lands there. |
| Desktop app (0.2) | `apps/host/src/lib/native-catalog.ts`, `native-fit.ts`, `native-names.ts`, `strata_control/models.py` (`SUPPORTED_SELECTIONS`), `data/gguf_fingerprints.json`, `tools/strata_inspect.py` (`FORK_KNOWN`), `tools/quantscope.py` | The app's Models page shows `orca`, `mrad`, `rvn` under "Uncensored models" (`FORK_FAMILIES` / `FORK_EDITIONS`; upstream's page keeps only its own family ids and Flash-Next titles). The app bundles setup.py, `strata_mcp.py`, `iq_pack.py` and `data/`, so an app built from this fork installs them as setup does (the gated token: `HF_TOKEN` or `hf auth login`). "Use model files I already have" knows their GGUFs by name and by header (fingerprint rows read from Hugging Face; `quantscope` sends `HF_TOKEN` to huggingface.co for the gated repository). |
| File pins | `setup.py` (`REPO_PINS.update` below upstream's) | Every file the fork's families download has its size and SHA-256 at the `HF_REVISIONS` commit (upstream's 0.2 test: every downloaded file is pinned). |

Fork-only files (upstream has none of these, so they never conflict): `docs/FORK.md`,
`scripts/merge-upstream.sh`, `scripts/merge-upstream.ps1`, `tools/test_fork_mcp.py`,
`tools/test_fork_setup.py`, `apps/host/tests/fork-catalog.test.mjs`, `MoreSimpleStart.*`.

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
  (`low_ram_*`) stay on `MODELS`, and the low-RAM mode is off for families that carry their own `sizes`. The
  v0.1.43 merge's new one was the #967 images warning (`sizes[model].get("vision_untested")`); the ones inside
  `if sizes[model].get("budget")` and `if low_ram` cannot be reached by a fork family and stay on `MODELS`.
- **`setup.py` / HIP engine**: upstream's `build_engine_hip`, `get_prebuilt_hip` and `ensure_engine_for` write
  `ROOT / "engine"`; that is `engine_dir()` (`engine/` on Windows, `engine-linux/` on Linux; the HIP engine is
  never the CUDA 12 one, so it stays on the default toolkit 13).
- **`setup.py` / `download`**: the fork's `token=` is passed only for a gated model, so upstream's tests' fake
  `download(url, dst, what=None)` keeps working. `write_run_script` keeps upstream's three arguments (the
  vision suffix comes from the config's name).
- **`setup.sh`**: the environment is `.venv-linux`, not `.venv` (upstream's NixOS/`STRATA_UV` branch, `UV` and
  `--managed-python`, creates that same `.venv-linux`).
- **`docker-entrypoint.sh`**: the config is the name `setup.py` writes on Linux
  (`strata-linux-<tag>-vision|novision.json`), so the link in `/opt/strata` is `link="/opt/strata/$name"`, not
  upstream's `strata-$tag.json`: the fork's `installed_configs` accepts only `strata-linux-*` on Linux, so
  upstream's name would be ignored and the container would set the model up again at every start. Keep
  upstream's `ln -sfn`, its #1244 comment and its `Config: ...` lines.
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
- **`setup.py` / `download` (0.2)**: upstream's parallel ranged download (`_Source`, `_fetch_run`, `download_single`,
  `range_total`); the fork's `auth` header (the gated token) goes on every request of it, so each takes an `auth=None`.
- **`setup.py` / `home()` (0.2)**: upstream writes configs, logs and run scripts to `home()` (the app's install folder,
  else `ROOT`); the fork's names stay, in `home()`: `home() / f"{stem}.json"`, `model_config_files(home())` in
  `installed_configs`. Engines stay in `engine_dir()` (the app's own engine is a component, not in the checkout).
- **Images on by default (0.2)**: a test that writes an earlier config for a case whose images are now on names it
  `-vision` (`test_setup_config.py`, `test_user_args.py`); the golden logs follow the config (a case's `log` differs
  from upstream's only by the `-vision|novision` suffix).
- **Orca IQ3_XXS**: upstream's `docs/ORCA.md` is the manual workflow the fork's `orca` family runs; its docs and the
  translated READMEs say the installer and the app do it here.

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
| v0.1.43 | `5f0852c7` | engine 0.1.41-0.1.43: the HIP layer split, the route-resident swap, the canonical Q2_0 pack, the #1892 NaN guard, the #1244 Docker config link, 2307 files. `setup.py` (2 conflicts): upstream's four-part hotfix version (`[:4]`) with the fork's `sizes[model]` for `need_engine`, and `sizes[model]` for the new #967 `vision_untested` read in `main()` (no conflict, but `MODELS[model]` there would raise `KeyError` for a fork family); `setup.sh`: the fork's `.venv-linux` with upstream's uv/NixOS branch in one place; `docker-entrypoint.sh`: the fork's config name for the `/opt/strata` link (`link="/opt/strata/$name"`) with upstream's `ln -sfn` and `Config: ...` lines; `README.md` and `docs/DETAILS.md`: both sides kept (the dual-boot text and the uncensored table, upstream's `BENCH.bat` line and its several-keys text). The other files the fork touches merged cleanly and were checked one by one |
| v0.1.45 | `4ccc2fbc` | engine 0.1.44-0.1.45: Unsloth's five files and the RAM-resident mode (`unsloth_budget`), the NVFP4 ready-made packs (`ready`), Qwen3.6 (`OWN_PATH_FAMILIES`), the macOS standalone Python in `setup.sh`, 579 commits. `setup.py` (11 conflicts): `engine_dir()`/`build_dir()`/`CFG_PREFIX` kept beside upstream's `KQ_LIB`; `choices_from_config` strips the fork's `-vision|novision` and then runs upstream's Qwen3.6 branch; `--family`/`--model` choices list the fork's families and the Qwen3.6 sizes; the size menu uses upstream's `size_tag` / `recommended_size`, but a family with its own `sizes` keeps the fork's pick (`recommended_size` reads `MODELS[m]`, which has no `IQ2_M`); `sizes[model]` for upstream's `MODELS[model].get("budget")` (budget block, `need_engine`) and for the new `license` read (no `KeyError` for a fork family); the fork's gated `token=` and `compat_bf16` kept inside upstream's `ready` / `check_shards` changes; `cfg_path` is the fork's `{stem}.json`, with upstream's `a.images` write kept. `tools/strata_mcp.py`: upstream's "slow here" verdict with the fork's `"sizes" not in fd`, and `sz.get("budget")` with upstream's `unsloth_budget` check. Tests: golden `-iq3_s-novision.log` and `-vision.log` for the Unsloth cases that now turn images on, `test_setup_menu.py` lists `orca`, `mrad`, `rvn` after `swift`, `test_fork_setup.py` skips upstream's ready-made families (no `file` pattern). Known gap: Qwen3.6's config is `strata-qwen36-*.json` without `CFG_PREFIX`, so on Linux `installed_configs` does not list it |
| v0.2.1.1 | `fdceb4ec` | app 0.2.1 (the Tauri desktop app, Strata Control, the Terminal), engine 0.1.46 as a signed component package, parallel downloads, Gyro, 890 commits. `setup.py` (19 conflicts): the fork's token through upstream's parallel download, `home()` with the fork's config names, `engine_dir()` with upstream's `acquisition.require`, `sizes[model]` for the new `mem_total` / `ready` / `gyro` / `--ple-io` reads; `tools/strata_mcp.py` (11): upstream's Qwen3.6 `own_size` before the fork's `size_table`; `docker-entrypoint.sh` removed (upstream's `docker/entrypoint.sh` globs `strata-*.json`, which the fork's names match). The fork's files got `REPO_PINS`. Tests: 8 failures and 5 errors, all also on pristine upstream on this PC. The engine package `strata-engine-0.1.46-windows-x86_64-cuda-cuda13.tar.gz` (SHA-256 checked against the release's `SHA256SUMS.txt` and its manifest) went into `engine/` with `setup.install_unpacked` (0.1.45 in `engine/.previous`): the checkout's `--update-engine` looks for `strata-windows-x64.zip`, which the 0.2 releases no longer carry |
| v0.2.4.2 | (this merge) | app 0.2.3, Terminal 0.2.4.2, Control 0.1.49.2: DeepSeek-V4.1-Flash (`MULTI_FILE_FAMILIES`, its own engine package), the download repair (`repair_download`), Unsloth UD-Q5/Q6_K_XL and Q8_0, Swift IQ3_S, the restyled READMEs, 654 commits. `setup.py` (10 conflicts): the fork's `auth` through upstream's new `_repair_piece` / `repair_download` and the stricter `download_single`; `hip_probe_path` and the DeepSeek engine folder are `engine_dir()`, not `ROOT / "engine"` (no conflict for the second); `choices_from_config` strips `-vision|novision` before upstream's DeepSeek branch; `--family`/`--model` choices list the fork's sizes and DeepSeek's; the new disk-space check with the fork's gated `token` after it; the fork's config name for `saved_draft_vocab` inside upstream's `experts.bin` check; the fork's `sys.executable` hint in `ensure_build_tools`. `strata_control/models.py` and `data/gguf_fingerprints.json`: both sides' new sizes. README: the fork's uncensored table is its own section (upstream dropped the bullet list it sat in); the translations keep the fork's Orca line as a paragraph before the DeepSeek section. Tests: `tools` 1232 run, every failure also on pristine upstream on this PC or a timing failure that passes alone; `tests/terminal` needs `textual` (11 modules not collected here). Known gap: DeepSeek's config is `strata-<tag>.json` without `CFG_PREFIX` (as Qwen3.6's) |

`git rerere` replayed the recorded `ROOT / "engine"` -> `engine_dir()` and `ROOT / "build"` -> `build_dir()`
renames onto upstream's new code in the v0.1.38 merge, so those hunks merged without a conflict. The Python
tests run with the fork's `.venv` (470 before the v0.1.43 merge; 467 since v0.1.40.1; 294 before it): 685 now,
673 pass, 10 skip, 0 errors (the two errors of the merge itself were upstream's new `tools/strata_bench.py`
on this PC's cp949 locale - the one-line fix is the Windows-locale row above).

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