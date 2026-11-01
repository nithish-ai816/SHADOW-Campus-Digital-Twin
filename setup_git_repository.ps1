# SHADOW - Git Repository Staged Commit and Push Script (PowerShell)
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  SHADOW - Git Repository Staged Commit and Push Tool" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

# 1. Initialize
Write-Host "[1/8] Initializing local Git repository..." -ForegroundColor Yellow
git init
git branch -M main

# 2. Commit 1: Original
Write-Host "`n[2/8] Commit 1: Initial Baseline (Your Original SHADOW Files)..." -ForegroundColor Yellow
git add app_new.py shadow_data.json shadow_users.json assets/campus_map.jpg assets/shadow_mark.png assets/admin_avatar.png
git commit -m "Initial commit: Original SHADOW campus energy monitoring dashboard"

# 3. Commit 2: Backend Engines
Write-Host "`n[3/8] Commit 2: Modular Architecture and Core AI Engines..." -ForegroundColor Yellow
git add camera_sources.py energy_waste.py device_integration.py test_suite.py
git commit -m "feat(core): Modular architecture for camera AI, energy waste estimation, and device integration"

# 4. Commit 3: Video Assets
Write-Host "`n[4/8] Commit 3: Classroom Surveillance Footage and Playable Video Assets..." -ForegroundColor Yellow
git add assets/*.mp4 assets/*.webp
git commit -m "feat(assets): 10 sample classroom surveillance recordings and browser-playable WebP animations"

# 5. Commit 4: Theme & Data
Write-Host "`n[5/8] Commit 4: Dark Theme Configuration and Digital Twin Dataset..." -ForegroundColor Yellow
git add .streamlit/config.toml shadow_ai_data.json
git commit -m "style(theme): Enforce campus dark theme and initialize digital twin room dataset"

# 6. Commit 5: AI Engine
Write-Host "`n[6/8] Commit 5: High-Tech CCTV Surveillance AI HUD..." -ForegroundColor Yellow
git add camera_ai.py
git commit -m "feat(ai): CCTV surveillance overlay with Identified/Unknown target badges, head boxes, and HUD"

# 7. Commit 6: Upgraded Dashboard
Write-Host "`n[7/8] Commit 6: Upgraded SHADOW System Dashboard..." -ForegroundColor Yellow
git add app_shadow_ai.py
git commit -m "feat(ui): Upgraded SHADOW dashboard with sidebar toggle, faculty routing, and external shutdown controls"

# 8. Commit 7: Local Receiver & Docs
Write-Host "`n[8/8] Commit 7: Local SMTP Test Receiver and Documentation..." -ForegroundColor Yellow
git add local_email_receiver.py .gitignore requirements.txt README.md setup_git_repository.bat setup_git_repository.ps1
git commit -m "docs: Add local SMTP receiver, hardware schematics, and complete system documentation"

Write-Host "`n================================================================" -ForegroundColor Green
Write-Host "  All Staged Commits Created Successfully!" -ForegroundColor Green
Write-Host "================================================================`n" -ForegroundColor Green

git log --oneline --graph

Write-Host "`nTo push this repository to GitHub:" -ForegroundColor Cyan
Write-Host "1. Create a new empty repository on https://github.com/new"
Write-Host "2. Run:"
Write-Host "   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor White
