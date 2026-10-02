"""Pick an installed model and start it: lists the start scripts setup.py wrote for this OS
(run-<model>-vision|novision.bat on Windows, .sh elsewhere) and runs the chosen one.

Run it through MoreSimpleStart.bat (Windows) or ./MoreSimpleStart.sh (Linux); nothing is installed or changed."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = sys.platform == "win32"
NAME = re.compile(r"^run-(.+)-(vision|novision)\.(bat|sh)$")


def scripts():
    """(path, model, vision|novision) for each start script of this OS, by model name."""
    ext = "bat" if WIN else "sh"
    found = []
    for p in sorted(ROOT.glob(f"run-*.{ext}")):
        m = NAME.match(p.name)
        if m:
            found.append((p, m.group(1), m.group(2)))
    return found


def main() -> int:
    found = scripts()
    if not found:
        print("No start script found (run-<model>-vision|novision." + ("bat" if WIN else "sh") + ").")
        print("Install a model first: " + ("START-HERE.bat" if WIN else "./setup.sh"))
        return 1
    print("Strata - installed models:")
    for i, (_, model, vision) in enumerate(found, 1):
        print(f"  {i}) {model}  [{vision}]")
    while True:
        pick = input(f"Which one? [1-{len(found)}, q to quit] (1): ").strip().lower() or "1"
        if pick == "q":
            return 0
        if pick.isdigit() and 1 <= int(pick) <= len(found):
            break
        print(f"  enter a number from 1 to {len(found)}")
    script = found[int(pick) - 1][0]
    print(f"Starting {script.name} ...")
    cmd = ["cmd", "/c", str(script)] if WIN else ["sh", str(script)]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nstopped.")
        sys.exit(1)
