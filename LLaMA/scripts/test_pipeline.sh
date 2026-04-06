#!/bin/bash
# Quick test script for the Muon & RMNP pipeline

echo "===== Muon & RMNP Pipeline Test ====="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project directory: $PROJECT_DIR"
echo

# Test 1: Check if main script exists
echo "Checking main training script..."
if [[ -f "$PROJECT_DIR/torchrun_main.py" ]]; then
    echo "OK: torchrun_main.py exists"
else
    echo "FAIL: torchrun_main.py not found"
fi

# Test 2: Check if training scripts exist
echo
echo "Checking training scripts..."
scripts=(
    "train_universal.sh"
    "train_muon_60m.sh"
    "train_muon_135m.sh"
    "train_muon_350m.sh"
    "train_RMNP_60m.sh"
    "train_RMNP_135m.sh"
    "train_RMNP_350m.sh"
)

for script in "${scripts[@]}"; do
    if [[ -x "$PROJECT_DIR/scripts/$script" ]]; then
        echo "OK: $script exists and is executable"
    else
        echo "FAIL: $script not found or not executable"
    fi
done

# Test 3: Check if config files exist
echo
echo "Checking model config files..."
configs=("llama_60m.json" "llama_135m.json" "llama_350m.json" "llama_1b.json")
for config in "${configs[@]}"; do
    if [[ -f "$PROJECT_DIR/configs/$config" ]]; then
        echo "OK: $config exists"
    else
        echo "FAIL: $config not found"
    fi
done

# Test 4: Check if optimizer files exist
echo
echo "Checking optimizer files..."
if [[ -f "$PROJECT_DIR/optimizers/muon_optimizer.py" ]]; then
    echo "OK: muon_optimizer.py exists"
else
    echo "FAIL: muon_optimizer.py not found"
fi

if [[ -f "$PROJECT_DIR/optimizers/RMNP_optimizer.py" ]]; then
    echo "OK: RMNP_optimizer.py exists"
else
    echo "FAIL: RMNP_optimizer.py not found"
fi

# Test 5: Check syntax of main script
echo
echo "Checking Python script syntax..."
python3 -m py_compile "$PROJECT_DIR/torchrun_main.py" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "OK: torchrun_main.py syntax is valid"
else
    echo "FAIL: torchrun_main.py has syntax errors"
fi

# Test 6: Dry run of universal script help
echo
echo "Testing universal script help..."
if "$PROJECT_DIR/scripts/train_universal.sh" --help > /dev/null 2>&1; then
    echo "OK: train_universal.sh help works"
else
    echo "FAIL: train_universal.sh help failed"
fi

echo
echo "===== Test complete ====="
echo
echo "If all items show OK, the pipeline is ready to use."
echo
echo "Examples:"
echo "cd $PROJECT_DIR"
echo "./scripts/train_muon_60m.sh          # Train Muon 60M model"
echo "./scripts/train_RMNP_135m.sh         # Train RMNP 135M model"
echo
echo "See full documentation: cat README.md"
