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

            TARBALL_URL="https://ollama.com/download/ollama-linux-${ARCH}.tgz"

            info "Downloading Ollama (linux-${ARCH}) to ${INSTALL_DIR} …"
            info "URL: ${TARBALL_URL}"
            if curl -fL --progress-bar "$TARBALL_URL" | tar -xzf - -C "$INSTALL_DIR" --strip-components=1 bin/ollama 2>/dev/null \
               || curl -fL --progress-bar "$TARBALL_URL" | tar -xzf - -C "$INSTALL_DIR" 2>/dev/null; then
                # The tarball extracts bin/ollama and lib/ollama/ — move binary if nested
                if [[ -f "${INSTALL_DIR}/bin/ollama" && ! -f "${INSTALL_DIR}/ollama" ]]; then
                    mv "${INSTALL_DIR}/bin/ollama" "${INSTALL_DIR}/ollama"
                fi
                chmod +x "${INSTALL_DIR}/ollama"
                success "Ollama installed at ${INSTALL_DIR}/ollama"

                # Ensure ~/.local/bin is on PATH
                if ! echo "$PATH" | grep -q "${INSTALL_DIR}"; then
                    _add_export "PATH" "${INSTALL_DIR}:\$PATH"
                    export PATH="${INSTALL_DIR}:${PATH}"
                    warn "Added ${INSTALL_DIR} to PATH in ${PROFILE}"
                fi

                # Set model storage location
                _add_export "OLLAMA_MODELS" "$model_dir"
                export OLLAMA_MODELS="$model_dir"

                OLLAMA_INSTALLED=1
            else
                error "Download failed. Try manually:"
                echo "  curl -fL ${TARBALL_URL} | tar -xzf - -C ${INSTALL_DIR}"
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
