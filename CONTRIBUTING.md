# Contributing to DriftGuard

Thank you for your interest in contributing to **DriftGuard**! This document provides guidelines and instructions on how to set up the development environment, run tests, adhere to code styles, and submit Pull Requests.

---

## 🚀 Quickstart Development Setup

To contribute changes to DriftGuard, first clone the repository and set up a Python virtual environment:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/imsanjoykb/agents-driftguard.git
   cd agents-driftguard
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Activate on Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies in Development/Editable Mode:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

---

## 🧪 Testing

We require all contributions to pass our existing test suite. Please run tests before committing code changes:

- **Run all tests:**
  ```bash
  pytest agents_driftguard/tests/ -v
  ```
- **Run with coverage (optional):**
  ```bash
  pytest --cov=agents_driftguard agents_driftguard/tests/
  ```

---

## 🎨 Coding Style & Linting

We enforce strict formatting rules using **Black** and **Ruff**. You can format and lint your code using the following commands:

- **Code Formatting:**
  ```bash
  black agents_driftguard/
  ```
- **Code Linting:**
  ```bash
  ruff check agents_driftguard/
  ```

Alternatively, you can install the **pre-commit** hooks to run these automatically on every commit:
```bash
pre-commit install
```

---

## 📥 Pull Request (PR) Guidelines

When submitting a pull request, please make sure to follow these rules:

1. **Keep PRs Focused:** Submit smaller, single-topic pull requests rather than giant code dumps.
2. **Write Unit Tests:** If you add a new feature or fix a bug, please write tests for it in the `agents_driftguard/tests/` folder.
3. **Update Documentation:** Ensure the README or internal code docstrings are updated to match your changes.
4. **Clean Git History:** Rebase your branch against the latest `master` and ensure commits are clean and well-named.
