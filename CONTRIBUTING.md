# How to Contribute to Safetensors

Everyone is welcome to contribute, and we value everybody's contribution. Code contributions are not the only way to help the community: answering questions, helping others, and improving the documentation are also immensely valuable.

However you choose to contribute, please be mindful and respect our [code of conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Fixing Bugs](#fixing-bugs)
- [Submitting a Feature Request](#submitting-a-feature-request)
- [Implementing a Feature](#implementing-a-feature)
- [Asking for Help](#asking-for-help)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Prerequisites](#prerequisites)
  - [Rust Core](#rust-core)
  - [Python Bindings](#python-bindings)
  - [Running Tests](#running-tests)
- [Style Guide](#style-guide)
  - [Rust](#rust)
  - [Python](#python)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Review Process](#review-process)

---

## Ways to Contribute

There are many ways to contribute to Safetensors:

- **Fix outstanding issues** in the [issue tracker](https://github.com/huggingface/safetensors/issues), especially those tagged [`good first issue`](https://github.com/huggingface/safetensors/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or [`help wanted`](https://github.com/huggingface/safetensors/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).
- **Submit bug reports or feature requests** by [opening a new issue](https://github.com/huggingface/safetensors/issues/new/choose).
- **Improve documentation** — if anything is unclear, incomplete, or missing, a PR is always appreciated.
- **Write tests** — more test coverage means more confidence in correctness.
- **Help others** by answering questions in [issues](https://github.com/huggingface/safetensors/issues).

---

## Fixing Bugs

If you find a bug and want to fix it yourself, please do! Start by opening an issue to describe the bug and let the maintainers know you're working on it — this prevents duplicate effort.

> **Do you have a security concern?** Please do **not** open a public issue. Instead, follow our [security policy](SECURITY.md) to report it responsibly.

When submitting a bug fix:

1. Link the relevant issue in your pull request description.
2. Include a test that would have caught the bug.
3. Keep the fix focused — avoid mixing unrelated changes in the same PR.

---

## Submitting a Feature Request

We love ideas! Before opening a feature request issue, please **search existing issues** to see if it's already been discussed.

A good feature request explains:
- The **problem** you're trying to solve (not just the proposed solution).
- Why you believe this belongs in the core library rather than userspace.
- Whether you're willing to implement it yourself.

Because safetensors deliberately aims to remain a **simple, minimal format**, we are conservative about adding new features. Features that add significant complexity to the spec or break backward compatibility will need a strong justification.

---

## Implementing a Feature

> We strongly recommend opening an issue or discussion before investing significant time implementing a new feature, especially if it touches the binary format specification. The maintainers can give early feedback and confirm the feature is likely to be accepted.

For **small, clearly-scoped improvements** (e.g., an ergonomic API addition in the Python bindings), you're welcome to open a PR directly.

For **spec-level changes** (new dtypes, changes to the header format, etc.), please open an issue first — these changes require careful design review and community input.

---

## Asking for Help

Don't hesitate to ask questions! You can:

- Open a [GitHub Discussion](https://github.com/huggingface/safetensors/discussions)
- Tag your StackOverflow question with `safetensors`

---

## Setting Up Your Development Environment

### Prerequisites

The safetensors repository contains a Rust core library and Python bindings (with optional support for PyTorch, NumPy, TensorFlow, JAX, and PaddlePaddle). You'll need:

- **Rust** (stable toolchain) — [install via rustup](https://rustup.rs/)
- **Python 3.10+**
- `pip` and optionally a virtual environment manager
- [optional] we strongly recommend installing uv for python management, but that's up to you!

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update

# Clone the repo
git clone https://github.com/huggingface/safetensors.git
cd safetensors
```

### Rust Core

The core library lives in `safetensors/`. To build and test it:

```bash
cd safetensors

# Build
cargo build

# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Check formatting and lints
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

### Python Bindings

The Python bindings live in `bindings/python/`. They use [PyO3](https://pyo3.rs/) via `maturin`.

```bash
cd bindings/python

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install setuptools-rust maturin
pip install -e ".[dev]"

# Build the Rust extension in-place (needed after changing Rust code)
pip install -e .
# or, for faster incremental builds during development:
maturin develop
```

> **Tip:** After every change to the Rust code, re-run `maturin develop` (or `pip install -e .`) before running Python tests — otherwise you'll be testing stale compiled code. You can also add the `--release` flag when building with maturin for a proper "release" build, which will be faster than the dev build.

#### Installing ML Framework Dependencies

Install the frameworks you want to test against:

```bash
# PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NumPy
pip install numpy

# TensorFlow
pip install tensorflow-cpu

# JAX
pip install jax

# PaddlePaddle
pip install paddlepaddle
```

### Running Tests

#### Rust Tests

```bash
# From the repo root
cargo test                        # core library tests
cargo test --manifest-path safetensors/Cargo.toml --all-features
```

#### Python Tests

```bash
cd bindings/python

# Run all Python tests
pytest tests/

# Run a specific test file
pytest tests/test_torch_serialization.py

# Run with verbose output
pytest -v tests/

# Run and stop at first failure
pytest -x tests/
```

#### Testing Across Frameworks

If you've installed multiple frameworks, run the full test suite to make sure your change doesn't break any of them:

```bash
pytest tests/ -v --tb=short
```

---

## Style Guide

Please match the existing code style. We use automated formatters and linters — a PR that fails CI checks will need to be fixed before it can be reviewed.

### Rust

We follow standard Rust conventions as enforced by the Rust toolchain:

```bash
# Format code
cargo fmt

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

- Use `cargo fmt` before every commit.
- Fix all `clippy` warnings — warnings are treated as errors in CI.
- Write documentation comments (`///`) for all public types and functions.
- Add unit tests inline in the same file (`#[cfg(test)]`).

### Python

We use [`ruff`](https://github.com/astral-sh/ruff) for both linting and formatting:

```bash
pip install ruff

# Check linting
ruff check bindings/python/

# Format code
ruff format bindings/python/

# Check formatting without modifying files
ruff format --check bindings/python/
```

Additional guidelines:

- Public Python APIs should have docstrings in [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- Type annotations are encouraged for all new code.
- Keep the Python surface area small and consistent — prefer matching the existing API style over introducing new patterns.

---

## Submitting a Pull Request

When you're ready to submit your contribution:

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b your-username/fix-brief-description
   ```

2. **Make your changes**, following the style guide above.

3. **Add or update tests** for any code you change. PRs that reduce test coverage are unlikely to be accepted.

4. **Update documentation** if your change affects user-facing behavior (docstrings, `README.md`, `docs/`).

5. **Run the full test suite** locally and make sure everything passes before pushing.

6. **Open a pull request** against the `main` branch. In the description:
   - Summarize **what** you changed and **why**.
   - Link any related issues (e.g., `Closes #123` or `Fixes #456`).
   - Mention any areas of the code you're uncertain about or that you'd like reviewers to pay special attention to.
   - We accept PRs that are made with the help of an AI model, but would appreciate if you
     fill out the AI model-related fields in the description.

7. **Sign your commits** if your organization requires it (optional otherwise). All commits merged into `main` should have a clean history.

> **Work in progress?** Open a [Draft PR](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to share early progress and get feedback before the implementation is complete.

---

## Review Process

We aim to review PRs within a few business days. The review process typically works like this:

1. A maintainer will be assigned to review your PR.
2. They may request changes or ask clarifying questions — please respond to comments as promptly as you can.
3. Once all feedback is addressed and CI passes, a maintainer will approve and merge the PR.

**A few things that speed up the review:**

- Keep PRs small and focused on a single concern. Large, sprawling PRs take much longer to review.
- Write a clear, descriptive PR description — it helps reviewers understand intent and context without having to ask.
- Don't resolve review comments yourself (leave that to the reviewer) so the conversation thread stays readable.
- Be patient and respectful. Maintainers are often juggling many things at once.

We appreciate every contribution, big or small. Thank you for taking the time to make Safetensors better! 🤗