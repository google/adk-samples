#!/bin/bash
#
# python-checks.sh: Runs formatting and linting checks on Python files based on arguments.
#
set -e # Exit immediately if a command exits with a non-zero status.

# --- Function Definitions ---

show_help() {
    echo "
Usage: ./python-checks.sh <FLAG> <PATH>

This script runs Python formatting (Black, iSort) and linting (Flake8) checks 
on a specified relative path.

Arguments:
  1. <FLAG>      : The action to perform (--run-black, --run-isort, --run-lint, --run-all).
  2. <PATH>      : Relative path to the folder (Must start with agents/ or notebooks/).

Examples:
  ./python-checks.sh --run-black agents/academic-research
  ./python-checks.sh --run-isort agents/academic-research
  ./python-checks.sh --run-lint agents/academic-research
  ./python-checks.sh --run-all notebooks/evaluation
  ./python-checks.sh --help
"
    exit 0
}

run_black() {
    echo -e "\n--- Running Black Formatting Check ---"
    local path="$1"
    if [ -n "$path" ]; then
        # Note: If you eventually use nbqa for notebooks, add that logic inside this if block
        if [[ "$path" == notebooks/* ]]; then
            black --check --diff "$path"
        else
            black --check --diff "$path"
        fi
        echo -e "Black check passed for: $path"
    fi
}

run_isort() {
    echo -e "\n--- Running iSort Import Check ---"
    local path="$1"
    if [ -n "$path" ]; then
        if [[ "$path" == notebooks/* ]]; then
            isort --profile black --check-only --diff "$path"
        else
            isort --profile black --check-only --diff "$path"
        fi
        echo -e "iSort check passed for: $path"
    fi
}

run_lint() {
    echo -e "\n--- Running Flake8 Linting Check ---"
    local path="$1"
    if [ -n "$path" ]; then
        if [ -d "$path" ]; then
            flake8 "$path"
            echo -e "Flake8 check passed for: $path"
        else
            echo "Error: Directory '$path' not found."
            exit 1
        fi
    fi
}

check_and_install_tools() {
    REQUIRED_TOOLS="black flake8 isort nbqa"
    TOOLS_MISSING=0
    
    if ! command -v black &> /dev/null; then
        TOOLS_MISSING=1
    fi
    
    if [ $TOOLS_MISSING -eq 1 ]; then
        echo "Installing required Python tools ($REQUIRED_TOOLS)..."
        python3 -m pip install $REQUIRED_TOOLS
    fi
    
    TOOLS_DIR=$(dirname "$(command -v black || echo "")")
    
    if [ -n "$TOOLS_DIR" ] && [ -d "$TOOLS_DIR" ]; then
        if [[ ":$PATH:" != *":$TOOLS_DIR:"* ]]; then
            export PATH="$TOOLS_DIR:$PATH"
            echo "Updated PATH to include Python tools at: $TOOLS_DIR"
        fi
    fi
}

# --- Argument Parsing & Validation ---

ACTION="$1"
TARGET_PATH="$2"

# 1. Handle Help
if [[ "$ACTION" == "--help" || "$ACTION" == "-h" ]]; then
    show_help
fi

# 2. Validate Action Flag
if [[ -z "$ACTION" ]]; then
    echo "Error: Missing argument. Use './python-checks.sh --help' for usage."
    exit 1
fi

if [[ "$ACTION" != "--run-all" && "$ACTION" != "--run-black" && "$ACTION" != "--run-isort" && "$ACTION" != "--run-lint" ]]; then
    echo "Error: Unknown flag '$ACTION'. Use --run-all, --run-black, --run-isort, or --run-lint."
    exit 1
fi

# 3. Validate Path Structure (Must start with agents/ or notebooks/)
if [[ -z "$TARGET_PATH" ]]; then
    echo "Error: Missing path argument for $ACTION."
    exit 1
fi

if ! [[ "$TARGET_PATH" =~ ^(agents/|notebooks/).* ]]; then
    echo "Error: Path '$TARGET_PATH' must start with 'agents/' or 'notebooks/'."
    exit 1
fi

if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: Directory '$TARGET_PATH' does not exist."
    exit 1
fi

# --- Main Execution ---

echo "Starting local Python code quality checks..."
check_and_install_tools

case "$ACTION" in
    --run-all)
        run_black "$TARGET_PATH"
        run_isort "$TARGET_PATH"
        run_lint "$TARGET_PATH"
        ;;
    --run-black)
        run_black "$TARGET_PATH"
        ;;
    --run-isort)
        run_isort "$TARGET_PATH"
        ;;
    --run-lint)
        run_lint "$TARGET_PATH"
        ;;
esac

echo -e "\n✅ Requested action ($ACTION) completed successfully for $TARGET_PATH!"