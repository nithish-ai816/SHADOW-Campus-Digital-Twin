@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo   SHADOW - Git Repository Staged Commit and Push Tool
echo ================================================================
echo.

cd /d "%~dp0"

echo [1/8] Initializing local Git repository...
git init
git branch -M main

echo.
echo [2/8] Commit 1: Initial Original Baseline (Your Original SHADOW Files)...
git add app_new.py shadow_data.json shadow_users.json assets\campus_map.jpg assets\shadow_mark.png assets\admin_avatar.png
git commit -m "Initial commit: Original SHADOW campus energy monitoring dashboard"

echo.
echo [3/8] Commit 2: Modular Architecture and Core AI Engines...
git add camera_sources.py energy_waste.py device_integration.py test_suite.py
git commit -m "feat(core): Modular architecture for camera AI, energy waste estimation, and device integration"

echo.
echo [4/8] Commit 3: Classroom Surveillance Footage and Playable Video Assets...
git add assets\*.mp4 assets\*.webp
git commit -m "feat(assets): 10 sample classroom surveillance recordings and browser-playable WebP animations"

echo.
echo [5/8] Commit 4: Dark Theme Configuration and Digital Twin Dataset...
git add .streamlit\config.toml shadow_ai_data.json
git commit -m "style(theme): Enforce campus dark theme and initialize digital twin room dataset"

echo.
echo [6/8] Commit 5: High-Tech CCTV Surveillance AI HUD...
git add camera_ai.py
git commit -m "feat(ai): CCTV surveillance overlay with Identified/Unknown target badges, head boxes, and HUD"

echo.
echo [7/8] Commit 6: Upgraded SHADOW System Dashboard...
git add app_shadow_ai.py
git commit -m "feat(ui): Upgraded SHADOW dashboard with sidebar toggle, faculty routing, and external shutdown controls"

echo.
echo [8/8] Commit 7: Local SMTP Test Receiver and Documentation...
git add local_email_receiver.py .gitignore requirements.txt README.md
git commit -m "docs: Add local SMTP receiver, hardware schematics, and complete system documentation"

echo.
echo ================================================================
echo   All 7 Staged Commits Created Successfully!
echo ================================================================
echo.
git log --oneline
echo.
echo To push this repository to GitHub:
echo 1. Create a new empty repository on https://github.com/new
echo 2. Run the following commands:
echo.
echo    git remote add origin https://github.com/^<YOUR_USERNAME^>/^<YOUR_REPO_NAME^>.git
echo    git push -u origin main
echo.
pause
