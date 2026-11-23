"""update-build-json.py - writes BUILD.json the way terminal/setup.py (strata launch's setup path) does.

Called by update-engines.sh / update-engines.ps1 after a built binary was deployed.  The hashes ("src", "vision_src")
and the version come from setup.py's own source_hash() / source_version(), so setup sees the engine as up to date.

  python update-build-json.py engine --dir engine       --kind strata --arch 120 --nvcc <nvcc.exe>
  python update-build-json.py vision --dir engine       --kind strata --vision cpu
  python update-build-json.py engine --dir engine-poly  --kind poly   --arch 120 --nvcc <nvcc.exe>
  python update-build-json.py vision --dir engine-poly  --kind poly   --vision cpu
  (--dry-run prints the JSON without writing)
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERMINAL = HERE / "terminal"


def load_setup():
    sys.path.insert(0, str(TERMINAL))
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("strata_setup", TERMINAL / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strata_setup"] = mod
    spec.loader.exec_module(mod)
    return mod


def archs_of(text):
    if text and text != "native":
        return sorted({int(x) for x in text.replace(",", ";").split(";") if x.strip()})
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return sorted({int(l.strip().replace(".", "")) for l in out.splitlines() if l.strip()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["engine", "vision"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--kind", choices=["strata", "poly"], required=True)
    ap.add_argument("--arch", default="native")
    ap.add_argument("--nvcc", default="")
    ap.add_argument("--vision", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    S = load_setup()
    eng = (HERE / a.dir)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}

    if a.what == "engine":
        archs = archs_of(a.arch)
        dirs = []
        if a.nvcc:
            bindir = Path(a.nvcc).parent       # the toolkit's own libraries (bin, bin/x64, lib64), as setup.py
            dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
        if a.kind == "poly":
            dirs += [d for d in S.cuda_lib_dirs(13) if d not in dirs]      # pip's, for an encoder kept beside it
        meta.update({"source": "local", "version": S.source_version(), "archs": archs, "cuda_dirs": dirs,
                     "toolkit": 13, "src": S.source_hash(S.ENGINE_SOURCES)})
        meta.setdefault("vision", "none")
        meta.setdefault("vision_src", None)
    else:
        meta.update({"vision": a.vision, "vision_src": S.source_hash(S.VISION_SOURCES)})

    text = json.dumps(meta, indent=1)
    if a.dry_run:
        print(text)
        return
    stamp.write_text(text, encoding="utf-8")
    print(f"  BUILD.json 갱신: {stamp}")


if __name__ == "__main__":
    main()
