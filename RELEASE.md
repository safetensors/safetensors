# Releasing safetensors

This document covers the full release process for the safetensors project. If anything here is unclear or out of date, please open a PR.

## What gets released

A single tag push triggers two releases:

- **Rust core crate** on [crates.io](https://crates.io/crates/safetensors), via `.github/workflows/rust-release.yml`
- **Python wheels + sdist** on [PyPI](https://pypi.org/project/safetensors), via `.github/workflows/python-release.yml` (with build-provenance attestations)

Both workflows trigger on tags matching `v*` (e.g. `v0.8.0`, `v0.8.0-rc.0`).

Conda packages are published separately via the [conda-forge/safetensors-feedstock](https://github.com/conda-forge/safetensors-feedstock), which picks up new PyPI releases automatically. No action is required from us.

The Rust crate and the Python binding share a version number. They could be versioned independently, but in practice they've always moved together.

## Pre-release checklist

1. **CI is green** on `main`, every platform, every Python version.
2. **Review the diff** since the last release:
   ```bash
   git log --oneline v0.7.0..main
   git diff v0.7.0..main --stat
   ```
3. **Identify breaking changes.** Anything that changes the public Rust or Python API should be called out in the release notes and reflected in the version bump per [semver](https://semver.org).
4. **Run benchmarks** if the release includes performance-sensitive work:
   ```bash
   cd safetensors && cargo bench
   ```
5. **Test against `transformers`.** It's the largest downstream consumer of safetensors.
   - At minimum, run the fast test suite:
     ```bash
     RUN_PIPELINE_TESTS=1 CUDA_VISIBLE_DEVICES=-1 pytest -sv tests/
     ```
   - For any significant release, run the full suite by rebasing these two PRs on the `transformers` repo:
     - [transformers#16708](https://github.com/huggingface/transformers/pull/16708) — builds docker images with `safetensors@main`
     - [transformers#16712](https://github.com/huggingface/transformers/pull/16712) — runs the full test suite
     If those PRs have drifted, ask the `transformers` team for the current entry point.
6. **Check the `transformers` version pin.** `transformers` pins `safetensors>=X,<Y` in its [setup.py](https://github.com/huggingface/transformers/blob/main/setup.py). If this release crosses the upper bound, coordinate with the `transformers` maintainers first.

## Cutting the release

We typically cut at least one release candidate before GA.

### 1. Update the dev version on `main`

`main` always carries a `-dev.0` version. Before branching, make sure it reflects the upcoming release:

```bash
# In both safetensors/Cargo.toml and bindings/python/Cargo.toml:
# e.g. 0.7.0-dev.0 → 0.8.0-dev.0
```

Update lockfiles accordingly:

```bash
cargo check
cd bindings/python && uv lock
```

Merge this to `main`.

### 2. Branch off `main`

```bash
git checkout main && git pull
git checkout -b git_v0.8.0
```

### 3. Set the release version on the branch

On the release branch, update both Cargo.toml files to the actual release version (drop the `-dev.0`):

```
0.8.0-dev.0 → 0.8.0
```

Update lockfiles again:

```bash
cargo check
cd bindings/python && uv lock
```

Commit:

```
Set version to 0.8.0
```

The Python version is not set in `pyproject.toml` directly — maturin reads it from `bindings/python/Cargo.toml` at build time.

### 4. Create the release on GitHub

Go to the [GitHub Releases page](https://github.com/huggingface/safetensors/releases) and draft a new release:

- **Choose the release branch** (e.g. `git_v0.8.0`)
- **Create a new tag** on publish: `v0.8.0`
- **Generate release notes** from the previous tag (e.g. `v0.7.0`)
- Add any notable highlights at the top of the generated notes

Structure for manual notes:

```markdown
## Breaking changes
- ...

## New features
- ...

## Bug fixes
- ...

## Internal / CI
- ...
```

Click **Publish release**. This creates the tag, which triggers the three release workflows.

### 5. Monitor the CI

The tag triggers two workflow runs in the [Actions tab](https://github.com/huggingface/safetensors/actions):

- **CI** (Python release) — builds wheels for every platform × Python version, uploads to PyPI. Can take up to over 30 minutes.
- **Rust Release** — `cargo publish` to crates.io. A few seconds to minutes.

The Python release uses `--skip-existing`, so re-running a partially failed workflow is safe. If `main` was green, the release builds should be too.

### 6. Verify

```bash
pip install safetensors==0.8.0
python -c "import safetensors; print(safetensors.__version__)"
```

Also check:
- [crates.io/crates/safetensors](https://crates.io/crates/safetensors)
- [pypi.org/project/safetensors](https://pypi.org/project/safetensors)

### 7. Bump `main` to the next dev version

After the release, open a PR on `main` setting both Cargo.toml files to the next dev version:

```
0.8.0-dev.0 → 0.9.0-dev.0
```

This keeps `main` clearly marked as unreleased.

## Release candidates

For releases with significant changes, cut an RC first:

- On the release branch, set the version to `0.8.0-rc.0` instead of `0.8.0`
- Tag as `v0.8.0-rc.0`, publish as a **pre-release** on GitHub
- Let it soak for a few days, ask downstream users to try it
- Fix regressions if needed (cut `-rc.1`, repeat)
- When ready, update the branch to `0.8.0` and publish the GA release

## Hotfixing a release branch

Sometimes a release or RC has a problem that needs fixing before GA (e.g. a CI matrix still referencing a dropped Python version). The fix goes on the release branch first, then gets upstreamed to `main`:

1. **Fix on the release branch.** Commit the fix directly to the release branch (e.g. `git_v0.8.0`).
2. **Re-tag and re-publish.** The procedure depends on what changed:
   - **CI-only fix** (no code changes in the published artifacts): delete the old tag, re-publish the same version. The wheels are identical so `--skip-existing` handles PyPI; only the failed jobs re-run.
     ```bash
     git tag -d v0.8.0-rc.0
     git push origin :refs/tags/v0.8.0-rc.0
     ```
     Go to the [Releases page](https://github.com/huggingface/safetensors/releases) — the previous release should appear as a draft. Re-publish it; this re-creates the tag on the latest commit and triggers the workflows.
   - **Code change**: bump the RC number in both Cargo.toml files (`0.8.0-rc.0` → `0.8.0-rc.1`). Leave the old tag in place (it's correct for its commit) and create a new release on GitHub with the new tag (`v0.8.0-rc.1`) pointing at the release branch.
3. **Cherry-pick to `main`.** Once the fix is validated on the release branch (CI green, artifacts published), cherry-pick it back to `main` via a feature branch:
   ```bash
   git checkout main && git pull
   git checkout -b fix/backport-<short-description>
   git cherry-pick <fix-commit-sha>
   git push origin fix/backport-<short-description>
   ```
   Open a PR from the feature branch so CI runs against `main` before merging. This ensures the fix isn't lost when the next release branches off.

## Secrets

The release workflows use two repository secrets:

- `PYPI_TOKEN_DIST` — PyPI API token for the `safetensors` project
- `CRATES_TOKEN` — crates.io API token

If a workflow fails with an auth error, the token has likely expired. Rotate it in the respective registry and update the repo secret. Worth checking before publishing the release.

## Testing release CI changes

If you're modifying the release workflows:

1. Comment out the upload steps (`maturin upload` / `cargo publish`).
2. Temporarily change the trigger to `push` on your branch.
3. Iterate until the artifacts build cleanly.
4. Revert both changes before merging.

## Troubleshooting

- **PyPI upload failed mid-matrix.** `--skip-existing` makes re-runs safe. Re-run from the Actions UI.
- **`cargo publish` says "already published".** You can't re-publish the same version. Bump to the next patch or pre-release and tag again.
- **Conda package not updated.** Conda packages are published via [conda-forge/safetensors-feedstock](https://github.com/conda-forge/safetensors-feedstock), which tracks PyPI. The feedstock's bot usually opens an auto-update PR within a day of the PyPI release; if it hasn't, ping the feedstock maintainers or open the PR yourself.
- **Workflow didn't trigger.** Verify the tag was pushed and that the name matches `v*`.
