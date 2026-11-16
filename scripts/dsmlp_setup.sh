#!/usr/bin/env bash
# Set up this project inside a DSMLP pod. Safe to re-run.
#
# Run INSIDE the pod (after `kubesh <pod-id>`):
#     curl -LsSf https://raw.githubusercontent.com/achinnam09/dsc180-quantization/ac/week8/scripts/dsmlp_setup.sh | bash
#   or, if the repo is already cloned:
#     bash ~/private/dsc180-quantization/scripts/dsmlp_setup.sh [branch]
#
# What it does:
#   1. Picks a persistent directory: ~/private if it exists (DSMLP pods mount
#      your real home there), otherwise ~. Everything that should outlive the
#      pod goes there: the repo, uv, Python, the uv cache and the Hugging Face
#      cache (model + LongBench).
#   2. Writes <persistent>/.kivi_env with PATH/HF_HOME/uv settings and sources
#      it from ~/.bashrc. Each new pod shell: `source ~/private/.kivi_env`
#      (done automatically if ~/.bashrc persists).
#   3. Clones the repo (or fetches) and checks out the branch (default ac/week8).
#   4. Installs uv if missing, runs `uv sync` (Python 3.13 + CUDA torch).
#   5. Checks the GPU: nvidia-smi, torch.cuda, bf16 support.

set -euo pipefail

BRANCH="${1:-ac/week8}"
REPO_URL="https://github.com/achinnam09/dsc180-quantization.git"

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

if [ -d "$HOME/private" ]; then PERSIST="$HOME/private"; else PERSIST="$HOME"; fi
REPO_DIR="$PERSIST/dsc180-quantization"
ENV_FILE="$PERSIST/.kivi_env"

step "Persistent directory: $PERSIST"
df -h "$PERSIST" | tail -1 || true

step "Environment file $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Written by scripts/dsmlp_setup.sh
export PATH="$PERSIST/.local/bin:\$PATH"
export HF_HOME="$PERSIST/hf_cache"
export UV_CACHE_DIR="$PERSIST/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PERSIST/.local/share/uv/python"
EOF
# shellcheck disable=SC1090
source "$ENV_FILE"
grep -qsF "source $ENV_FILE" "$HOME/.bashrc" || echo "source $ENV_FILE" >> "$HOME/.bashrc"
cat "$ENV_FILE"

step "Repository ($BRANCH)"
command -v git >/dev/null || die "git not found in this image"
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        die "$REPO_DIR has uncommitted changes to tracked files; commit/stash them, then re-run"
    fi
    git fetch origin
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH" || die "could not fast-forward $BRANCH; resolve manually in $REPO_DIR"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR" || die "clone failed (does branch $BRANCH exist on GitHub?)"
    cd "$REPO_DIR"
fi
git log --oneline -1

step "uv"
if ! command -v uv >/dev/null; then
    command -v curl >/dev/null || die "curl not found; cannot install uv"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$PERSIST/.local/bin" UV_NO_MODIFY_PATH=1 sh \
        || die "uv install failed"
fi
uv --version

step "uv sync (first time downloads Python 3.13 + CUDA torch: several GB, a few minutes)"
uv sync || die "uv sync failed"
uv cache prune --ci >/dev/null 2>&1 || true  # drop build leftovers to save quota
du -sh .venv "$UV_CACHE_DIR" 2>/dev/null || true

step "GPU check"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "WARNING: nvidia-smi not found (pod launched without -g 1?)"
fi
uv run python - <<'EOF' || die "torch cannot use the GPU; check the pod has a GPU and the driver supports this torch build"
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
name = torch.cuda.get_device_name(0)
print("GPU:", name, "| bf16 supported:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    print("WARNING: this GPU has no native bf16 (pre-Ampere, e.g. 1080/1080ti/2080ti). The model "
          "loads in bf16, so it will run slowly. Prefer a pod launched with -v a5000/a30/a100/l40s.")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul OK:", (x @ x).shape)
EOF

step "Done"
echo "Repo: $REPO_DIR"
echo "In each new shell in this pod:  source $ENV_FILE && cd $REPO_DIR"
echo "Then start the tests:"
echo "  mkdir -p logs && nohup bash scripts/run_kivi_tests.sh > logs/kivi_tests_\$(date +%m%d_%H%M).log 2>&1 &"
