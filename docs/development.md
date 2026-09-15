# Local development

## Branches

A branch is a named line of development. Work on `development/setup` is separate
from the commits on `main`. Uncommitted edits belong to the working directory;
commit them before switching branches so they do not follow you to another branch.

The initial setup branch was created with:

```bash
git switch -c development/setup
```

This command is already complete. To inspect the current branch and changes:

```bash
git branch --show-current
git status
git diff
```

`git diff` shows changes to tracked files. New files appear in `git status`; review
their contents too. After staging, `git diff --cached` shows the proposed commit.

## Install and check

Follow the Python 3.11 installation instructions in the root README. With the
environment activated, check the initial package:

```bash
python -c "import ct_mar, ct_mar.synthesis, ct_mar.inference; print(ct_mar.__version__)"
git diff --check
```

Run local checks with `python -m unittest discover -s tests`.
GPU simulation tests are skipped unless `CT_MAR_TEST_GPU=1` is set on a CUDA machine.
Model inference tests will follow its migration.

## Review, commit, and push

After reviewing the initial setup files:

```bash
git add README.md pyproject.toml .gitignore LICENSE.md THIRD_PARTY_NOTICES.md src configs examples docs weights third_party tests
git diff --cached
git commit -m "Set up synthesis and inference package structure"
git push -u origin development/setup
```

The push publishes the branch to your private repository. On GitHub, open a pull
request from `development/setup` into `main`. Review and merge when ready. Subsequent
work can use branches such as `feature/synthetic-generation` and `feature/inference`.

## Local artifacts

Keep datasets under `data/`, generated results under `outputs/`, and downloaded
weights under `checkpoints/`. These locations are ignored by Git. `.gitignore`
does not remove files already tracked; always inspect staged changes before a push.

## Migration checklist

- Keep generation and suppression independent, each with its own setup and example.
- Extract the full 3D generation workflow and its required dependencies.
- Extract both conditioned inference variants with compatible checkpoint loading.
- Preserve NIfTI spatial coordinates through preprocessing and output.
- Resolve the checkpoint's exact material vocabulary and embedding order.
- Add portable configurations, external weight locations, and runnable examples.
- Preserve upstream headers and record imported revisions and modifications.
- Select original-code and model-weight license terms before the public release.
