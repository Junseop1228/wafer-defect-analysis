$ErrorActionPreference = "Stop"

Write-Host "Committing phase2-features-sync..."
git add -A
git commit -m "feat: phase2-features-sync — Task 0 features.py Sync"

Write-Host "Checking out dev..."
git checkout dev

Write-Host "Merging stage2-eda..."
git merge --squash agent/stage2-eda
git commit -m "feat: stage2-eda — EDA"

Write-Host "Merging stage3-features-core..."
git merge --squash agent/stage3-features-core
git commit -m "feat: stage3-features-core — Core Features"

Write-Host "Merging stage3-features-extended..."
git merge --squash agent/stage3-features-extended
git commit -m "feat: stage3-features-extended — Extended Features"

Write-Host "Merging stage3-feature-matrix..."
git merge --squash agent/stage3-feature-matrix
git commit -m "feat: stage3-feature-matrix — Feature Matrix"

Write-Host "Merging stage3-gate1-check..."
git merge --squash agent/stage3-gate1-check
git commit -m "fix: stage3-gate1-check — Gate 1 Check"

Write-Host "Merging phase2-features-sync..."
git merge --squash agent/phase2-features-sync
git commit -m "feat: phase2-features-sync — Task 0 features.py Sync"

Write-Host "Checking out main..."
git checkout main

Write-Host "Merging dev to main..."
git merge dev

Write-Host "Tagging..."
git tag v1.0-gate1

Write-Host "Log..."
git log --oneline -10
