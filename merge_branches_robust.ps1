$ErrorActionPreference = "Continue"

git checkout dev
git reset --hard 94845c6

$branches = @(
    @("agent/stage2-eda", "feat: stage2-eda — EDA"),
    @("agent/stage3-features-core", "feat: stage3-features-core — Core Features"),
    @("agent/stage3-features-extended", "feat: stage3-features-extended — Extended Features"),
    @("agent/stage3-feature-matrix", "feat: stage3-feature-matrix — Feature Matrix"),
    @("agent/stage3-gate1-check", "fix: stage3-gate1-check — Gate 1 Check"),
    @("agent/phase2-features-sync", "feat: phase2-features-sync — Task 0 features.py Sync")
)

foreach ($b in $branches) {
    $branch = $b[0]
    $msg = $b[1]
    Write-Host "Merging $branch ..."
    git merge --squash $branch
    git checkout $branch -- .
    git add -A
    git commit -m $msg
}

git checkout main
git merge dev
git tag v1.0-gate1
git log --oneline -10
