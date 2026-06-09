#!/usr/bin/env bash
# setup_llm.sh — configure LLM providers for hprofiler AI analysis
#
# Usage: bash setup_llm.sh
#
# Installs Ollama to ~/.local/bin (no sudo needed — works on HPC clusters).
# Models are stored in ~/.ollama/models by default; override with OLLAMA_MODELS.

set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[setup]${NC} $*"; }
success() { echo -e "${GREEN}✔${NC}  $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✘${NC}  $*" >&2; }
ask()     { echo -e "${BOLD}$*${NC}"; }

echo
echo -e "${BOLD}${CYAN}━━━ hprofiler LLM setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# ── detect shell profile ──────────────────────────────────────────────────────
_profile() {
    local sh; sh=$(basename "${SHELL:-bash}")
    case "$sh" in
        zsh)  echo "$HOME/.zshrc" ;;
        fish) echo "$HOME/.config/fish/config.fish" ;;
        *)    echo "$HOME/.bashrc" ;;
    esac
}
PROFILE=$(_profile)

# ── helper: set/update an export line in the shell profile ───────────────────
_add_export() {
    local key="$1" val="$2"
    if grep -q "export ${key}=" "$PROFILE" 2>/dev/null; then
        sed -i "s|^export ${key}=.*|export ${key}=\"${val}\"|" "$PROFILE"
        warn "Updated ${key} in ${PROFILE}"
    else
        echo "export ${key}=\"${val}\"" >> "$PROFILE"
        success "Added ${key} to ${PROFILE}"
    fi
}

# ── detect architecture ───────────────────────────────────────────────────────
_arch() {
    case "$(uname -m)" in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        *)       echo ""; ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Ollama
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BOLD}Part 1: Ollama (local open-source models)${NC}"
echo

OLLAMA_INSTALLED=0

if command -v ollama &>/dev/null; then
    success "Ollama already installed: $(ollama --version 2>&1 | head -1)"
    OLLAMA_INSTALLED=1
else
    ask "Install Ollama? [Y/n]"
    read -r reply
    if [[ "${reply,,}" =~ ^(n|no)$ ]]; then
        warn "Skipping Ollama installation."
    else
        ARCH=$(_arch)
        if [[ -z "$ARCH" ]]; then
            error "Unsupported architecture: $(uname -m). Install Ollama manually from https://ollama.com/download"
        else
            INSTALL_DIR="${HOME}/.local/bin"
            mkdir -p "$INSTALL_DIR"

            # Where to store models — default ~/.ollama/models
            # On clusters with limited home quota, point this at a scratch/work directory.
            echo
            ask "Where should models be stored? [${HOME}/.ollama/models]"
            read -r model_dir
            model_dir="${model_dir:-${HOME}/.ollama/models}"
            mkdir -p "$model_dir"

            # Ollama v0.28+ ships as a .tar.zst bundle (includes GPU libs).
            # We fetch only the bin/ollama binary — the cluster already has CUDA/ROCm.
            GH_API="https://api.github.com/repos/ollama/ollama/releases/latest"
            TARBALL_NAME="ollama-linux-${ARCH}.tar.zst"

            # Resolve the versioned download URL via GitHub API
            info "Resolving latest Ollama release …"
            VERSIONED_URL=$(curl -fsSL --max-time 15 "$GH_API" 2>/dev/null \
                | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    for a in r.get('assets', []):
        if a['name'] == '${TARBALL_NAME}':
            print(a['browser_download_url'])
            break
except: pass
" 2>/dev/null || true)

            # Ollama needs bin/ollama + lib/ollama/llama-server + GPU libs.
            # Extract the full tarball to ~/.local/ with --strip-components=1 so that:
            #   bin/ollama              → ~/.local/bin/ollama
            #   lib/ollama/llama-server → ~/.local/lib/ollama/llama-server
            #   lib/ollama/lib*.so      → ~/.local/lib/ollama/lib*.so
            INSTALL_ROOT="${HOME}/.local"
            mkdir -p "${INSTALL_ROOT}/bin" "${INSTALL_ROOT}/lib"

            DOWNLOAD_OK=0

            _extract_zst() {
                # Tarball layout: bin/ollama  +  lib/ollama/llama-server  +  lib/ollama/*.so
                # Extract straight to ~/.local/ (no --strip-components) so that:
                #   bin/ollama              → ~/.local/bin/ollama
                #   lib/ollama/llama-server → ~/.local/lib/ollama/llama-server
                # Using --strip-components=1 would strip the first component (bin/ or lib/)
                # and create a name collision between the ollama binary and the ollama/ lib dir.
                local src="$1" dst="$2"
                if command -v zstd &>/dev/null; then
                    zstd -d --stdout "$src" | tar -xf - -C "$dst"
                else
                    tar --zstd -xf "$src" -C "$dst"
                fi
            }

            if [[ -n "$VERSIONED_URL" ]]; then
                TMP_TAR=$(mktemp /tmp/ollama_XXXXXX.tar.zst)
                info "Downloading Ollama from: ${VERSIONED_URL}"
                info "(~1.3 GB — includes llama-server + GPU libs)"
                if curl -fL --progress-bar "$VERSIONED_URL" -o "$TMP_TAR"; then
                    info "Extracting to ${INSTALL_ROOT} …"
                    if _extract_zst "$TMP_TAR" "$INSTALL_ROOT" \
                       && [[ -f "${INSTALL_ROOT}/bin/ollama" ]]; then
                        DOWNLOAD_OK=1
                        INSTALL_DIR="${INSTALL_ROOT}/bin"
                    else
                        warn "Extraction failed. tar version: $(tar --version | head -1)"
                        warn "Try extracting manually (see instructions below)."
                    fi
                fi
                rm -f "$TMP_TAR"
            fi

            if [[ "$DOWNLOAD_OK" == "0" ]]; then
                warn "Automatic install failed."
                echo
                echo -e "${BOLD}  ── Manual install — run directly on the cluster ─────────────${NC}"
                echo
                echo "  # Download the full tarball:"
                echo "  curl -fLO '${VERSIONED_URL:-https://github.com/ollama/ollama/releases/latest/download/${TARBALL_NAME}}'"
                echo
                echo "  # Extract to ~/.local/ (no --strip-components — tarball has bin/ + lib/):"
                echo "  mkdir -p ~/.local"
                echo "  zstd -d ${TARBALL_NAME} --stdout | tar -xf - -C ~/.local"
                echo "  # OR (if tar >= 1.31):"
                echo "  tar --zstd -xf ${TARBALL_NAME} -C ~/.local"
                echo
                echo "  # Verify:"
                echo "  ls ~/.local/bin/ollama ~/.local/lib/ollama/llama-server"
                echo -e "${BOLD}  ─────────────────────────────────────────────────────────────${NC}"
                echo
                ask "  Already extracted? Press Enter to continue, or Ctrl-C to abort:"
                read -r _dummy
                if [[ -f "${INSTALL_ROOT}/bin/ollama" ]]; then
                    DOWNLOAD_OK=1
                    INSTALL_DIR="${INSTALL_ROOT}/bin"
                fi
            fi

            if [[ "$DOWNLOAD_OK" == "1" ]]; then
                chmod +x "${INSTALL_DIR}/ollama"
                success "Ollama installed at ${INSTALL_DIR}/ollama"

                # Ensure ~/.local/bin is on PATH
                if ! echo "$PATH" | grep -q "${INSTALL_DIR}"; then
                    _add_export "PATH" "${INSTALL_DIR}:\$PATH"
                    export PATH="${INSTALL_DIR}:${PATH}"
                fi

                # Set model storage location
                _add_export "OLLAMA_MODELS" "$model_dir"
                export OLLAMA_MODELS="$model_dir"

                OLLAMA_INSTALLED=1
            else
                warn "Skipping Ollama — install the binary manually and re-run this script."
            fi
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Pull models
# ─────────────────────────────────────────────────────────────────────────────

PULLED_MODELS=()

if [[ "$OLLAMA_INSTALLED" == "1" ]]; then
    echo
    echo -e "${BOLD}Part 2: Open-source models${NC}"
    echo
    echo "  Choose models to pull (larger = smarter but slower and needs more VRAM/RAM):"
    echo
    echo "  [1] qwen2.5:1.5b      ~1 GB  — fits in 2 GB VRAM, fast on GPU"
    echo "  [2] llama3.1:8b       ~5 GB  — good tool use, needs 8 GB VRAM or 16 GB RAM"
    echo "  [3] qwen2.5:14b       ~9 GB  — better reasoning, needs 10 GB VRAM or 24 GB RAM"
    echo "  [4] qwen2.5:32b       ~20 GB — near-GPT-4 quality, needs 24 GB VRAM or 48 GB RAM"
    echo "  [5] deepseek-r1:8b    ~5 GB  — chain-of-thought reasoning"
    echo "  [6] none              skip"
    echo
    echo "  Enter numbers separated by spaces, or press Enter for [2]:"
    read -r model_choices
    model_choices="${model_choices:-2}"

    declare -A MODEL_MAP=(
        [1]="qwen2.5:1.5b"
        [2]="llama3.1:8b"
        [3]="qwen2.5:14b"
        [4]="qwen2.5:32b"
        [5]="deepseek-r1:8b"
        [6]="none"
    )

    # Start the Ollama daemon if not already running
    if ! ollama list &>/dev/null 2>&1; then
        info "Starting Ollama daemon in background …"
        # On clusters without systemd, run as a background process
        OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
        nohup ollama serve > "${HOME}/.ollama/serve.log" 2>&1 &
        OLLAMA_PID=$!
        info "Ollama daemon PID ${OLLAMA_PID} (log: ~/.ollama/serve.log)"
        _add_export "OLLAMA_HOST" "$OLLAMA_HOST"
        sleep 4
    fi

    for choice in $model_choices; do
        model="${MODEL_MAP[$choice]:-}"
        if [[ -z "$model" || "$model" == "none" ]]; then
            continue
        fi
        info "Pulling ${model} … (downloads model weights, may take several minutes)"
        if ollama pull "$model"; then
            success "Pulled ${model}"
            PULLED_MODELS+=("$model")
        else
            error "Failed to pull ${model}"
        fi
    done

    if [[ ${#PULLED_MODELS[@]} -gt 0 ]]; then
        DEFAULT_MODEL="${PULLED_MODELS[0]}"
        echo
        info "Default model → ${DEFAULT_MODEL}"
        _add_export "HPROFILER_LLM_PROVIDER" "ollama"
        _add_export "HPROFILER_LLM_MODEL" "$DEFAULT_MODEL"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Cloud API keys (optional)
# ─────────────────────────────────────────────────────────────────────────────

echo
echo -e "${BOLD}Part 3: Cloud API keys (optional)${NC}"
echo

ANTHROPIC_KEY=""
OPENAI_KEY=""

ask "Configure an Anthropic API key? [y/N]"
read -r reply
if [[ "${reply,,}" =~ ^y ]]; then
    ask "Paste your Anthropic API key (sk-ant-...):"
    read -rs ANTHROPIC_KEY; echo
    if [[ "$ANTHROPIC_KEY" == sk-ant-* ]]; then
        _add_export "ANTHROPIC_API_KEY" "$ANTHROPIC_KEY"
        _add_export "HPROFILER_LLM_PROVIDER" "anthropic"
        _add_export "HPROFILER_LLM_MODEL" "claude-sonnet-4-6"
        success "Anthropic key saved. Default model: claude-sonnet-4-6"
    else
        warn "Does not look like an Anthropic key (expected sk-ant-...) — not saved."
        ANTHROPIC_KEY=""
    fi
fi

echo
ask "Configure an OpenAI API key? [y/N]"
read -r reply
if [[ "${reply,,}" =~ ^y ]]; then
    ask "Paste your OpenAI API key (sk-...):"
    read -rs OPENAI_KEY; echo
    if [[ "$OPENAI_KEY" == sk-* ]]; then
        _add_export "OPENAI_API_KEY" "$OPENAI_KEY"
        success "OpenAI key saved. Default model: gpt-4o"
    else
        warn "Does not look like an OpenAI key (expected sk-...) — not saved."
        OPENAI_KEY=""
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo
echo -e "${BOLD}${CYAN}━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

[[ "$OLLAMA_INSTALLED" == "1" ]] && success "Ollama:   $(ollama --version 2>&1 | head -1)"
[[ ${#PULLED_MODELS[@]}   -gt 0 ]] && success "Models:   ${PULLED_MODELS[*]}"
[[ -n "$ANTHROPIC_KEY"         ]] && success "Anthropic key configured"
[[ -n "$OPENAI_KEY"            ]] && success "OpenAI key configured"

echo
info "Reload your shell:"
echo "    source ${PROFILE}"
echo
info "On the cluster, start Ollama before running hprofiler:"
echo "    ollama serve &"
echo "    python3 hprofiler analyze trace.hprofiler.json"
echo
info "To store models on scratch/work storage instead of home:"
echo "    export OLLAMA_MODELS=/scratch/\$USER/ollama-models"
echo "    ollama serve &"
echo
