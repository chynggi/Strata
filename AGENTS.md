# AGENTS.md

Version 0.2.16, engine 0.1.62. The public model families are
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

Strata has three parts:

- **The app** (`app/desktop`, the Tauri shell; `app/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.16-windows-x64-setup.exe`, `Strata-0.2.16-linux-x86_64.AppImage`, `Strata-0.2.16-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is setup from a source checkout, for developers. The release paths are the
  installer and `strata launch`.

## 포크 전용: 소스에서 엔진을 빌드하는 지침

Use this section when asked to build or rebuild an engine from source. The request is for a
binary built on the specified machine. Use the fork checkout on that machine and record its
path, branch and full commit (`git status --short --branch`, `git rev-parse HEAD`). Check the
target GPU and the installed compiler, toolkit and CMake before choosing build flags. If the
request names a model or config, check its engine executable to choose the target. Keep the
working tree and existing builds intact.

Build directly with CMake from the checkout root. CMake sources are in `engine/` since 0.2.16.
Use a separate build directory for each engine, toolkit and backend. On Windows with CUDA,
run from an x64 Visual Studio developer shell with the matching MSVC and CUDA toolkit.
The commands below use Ninja and `native` for a build on the target NVIDIA machine. Use an
explicit `CMAKE_CUDA_ARCHITECTURES` value for a build meant for another GPU. Quote each
`-Dkey=value` argument that contains spaces or semicolons.

- **Strata (`strata`).** Keep the default own-stack build. Leave `STRATA_BUILD_POLY` off and
  leave `STRATA_GGML_DIR` unset. Use the fork's source even for this target.

  ```sh
  cmake -S engine -B build-strata-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native
  cmake --build build-strata-cuda --target strata
  ```

- **Poly (`strata-poly`).** Enable `STRATA_BUILD_POLY` and pass a compatible full llama.cpp
  checkout as `STRATA_GGML_DIR`. Use an existing compatible checkout on the machine when
  available. For a new dependency checkout, use `LLAMA_CPP_COMMIT` from this revision's
  [terminal/setup.py](terminal/setup.py). The path is the llama.cpp root and must contain
  `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu`. The vendored
  `engine/third_party/strata-ggml` subset lacks the MMQ sources Poly needs.

  ```sh
  cmake -S engine -B build-poly-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native -DSTRATA_BUILD_POLY=ON "-DSTRATA_GGML_DIR=/path/to/llama.cpp"
  cmake --build build-poly-cuda --target strata-poly
  ```

- **Other backends.** Follow [HIP](docs/AMD_HIP.md), [SYCL](docs/INTEL_ARC.md), or
  [Metal](docs/MAC.md) for Strata. Poly supports CUDA and wave32 HIP in this fork. For Poly
  on HIP, use the matching HIP toolchain and GPU architecture, keep both Poly flags above,
  and build the `strata-poly` target. Use `build_engine_poly_hip` and
  `build_engine_poly_hip_win` in [terminal/setup.py](terminal/setup.py) as the reference for
  toolchain flags.
- **Vision.** Build `strata-vision` separately from `terminal/tools/vision` when requested.
  The default reader uses Strata's own vision code. For Poly's reader, configure its own
  build directory with `-DSTRATA_VISION_POLY=ON` and `-DLLAMA_DIR=<llama.cpp root>`. Use the
  compatible revision named by `VISION_LLAMA_CPP_COMMIT` for a new checkout. These builds
  produce binaries with the same filename. Keep their output paths distinct.

For an engine-only request, stop after building and checking the produced binary. Leave
installers, `START-HERE.bat`, `setup.sh`, and `strata launch` for setup requests. Those
flows can download release engines or models and write run configs. Deploy, replace an
installed engine, or restart a server only when the request includes that work. After
deployment, check the executable path in the run config.

Check the configure and build exit codes, locate the artifact in the selected build
directory, and invoke it by full path. Use `--version` for `strata` and `strata-vision`.
`strata-poly --help` prints usage and returns 2, so check the output as well as the exit
code. Report the machine, source commit, target, backend, GPU architecture, artifact path
and smoke-check result. Report model loading and inference only if they were run.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](tools/packaging/README.md) and [packaging/UPDATES.md](tools/packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `terminal/tests/terminal` (GitHub CI only on this PC). The app's UI
  tests run with `npm test` in `app/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `terminal/tools/test_setup_amd.py`, `terminal/tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
