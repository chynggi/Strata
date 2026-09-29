#!/bin/sh
# Fetch and merge upstream/main into this fork, reusing earlier conflict resolutions.
#
# The fork changes only a few files (see docs/FORK.md).  `git rerere` records how each conflict hunk was
# resolved and replays that resolution the next time the same hunk conflicts, so a repeated merge is mostly
# automatic.  Run this from anywhere inside the repository.
set -e
cd "$(dirname "$0")/.." || exit 1

git config rerere.enabled true
git config rerere.autoupdate true
git config merge.conflictStyle zdiff3

git fetch upstream --prune --tags
echo
echo "Merging upstream/main ..."
if git merge --no-edit upstream/main; then
  echo
  echo "Merged cleanly.  Push with: git push origin main"
else
  echo
  echo "Conflicts to resolve (git status).  Suggested order:"
  echo "  1. fix each file, keeping both sides where possible (docs/FORK.md lists the usual places)"
  echo "  2. git add <files> && git commit        # rerere records the resolutions"
  echo "  3. python -m unittest discover -s tools -p \"test_*.py\""
fi