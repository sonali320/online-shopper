# Antigravity Design System & Integration Guidelines

## Overview
This repository uses the **Antigravity** glassmorphic visual aesthetic and high-performance machine learning design patterns for Online Shoppers Purchasing Intention.

## Theme & Color Tokens
- **Background**: `#0a0d14` (Deep obsidian dark mode)
- **Card Background**: `rgba(18, 24, 38, 0.75)` with `backdrop-filter: blur(16px)`
- **Borders**: `rgba(255, 255, 255, 0.08)`
- **Primary / Accent**: `#6366f1` (Indigo electric)
- **Secondary Accent**: `#06b6d4` (Teal glow)
- **Success / High Intent**: `#10b981` (Emerald green)
- **Warning / Cart Abandoner**: `#f59e0b` (Amber)
- **Danger / No Purchase**: `#ef4444` (Ruby coral)
- **Text Primary**: `#f8fafc`
- **Text Muted**: `#94a3b8`

## Typography
- **Primary Font**: `Plus Jakarta Sans`, sans-serif
- **Monospace / Code**: `JetBrains Mono`, monospace

## Streamlit Integration
- Configuration located at `.streamlit/config.toml`
- Core frontend located at `streamlit.py`
- Batch launcher located at `run_app.bat`
- Dedicated Random Forest training script: `train_model.py`
