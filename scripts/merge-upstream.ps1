# Fetch and merge upstream/main into this fork, reusing earlier conflict resolutions.
#
# The fork changes only a few files (see docs/FORK.md).  git rerere records how each conflict hunk was
# resolved and replays that resolution the next time the same hunk conflicts, so a repeated merge is mostly
# automatic.  Run this from anywhere inside the repository.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

git config rerere.enabled true
git config rerere.autoupdate true
git config merge.conflictStyle zdiff3

git fetch upstream --prune --tags
Write-Host ""
Write-Host "Merging upstream/main ..."
git merge --no-edit upstream/main
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Merged cleanly.  Push with: git push origin main"
} else {
    Write-Host ""
    Write-Host "Conflicts to resolve (git status).  Suggested order:"
    Write-Host "  1. fix each file, keeping both sides where possible (docs/FORK.md lists the usual places)"
    Write-Host "  2. git add <files>; git commit          # rerere records the resolutions"
    Write-Host "  3. python -m unittest discover -s tools -p ""test_*.py"""
}