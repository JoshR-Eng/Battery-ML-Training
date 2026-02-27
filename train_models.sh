#!/bin/bash

# =================================================================
# Batch Model Training Script
# =================================================================
# 
# Usage:
#   ./train_models.sh                           # Train all models
#   ./train_models.sh LSTM GRU CNN-LSTM         # Train specific models
#   ./train_models.sh --name MyExperiment LSTM  # Custom experiment name
#
# Features:
#   - Trains multiple models sequentially
#   - Auto-updates config.yaml for each model
#   - Generates comparison table at the end
#   - Backs up config.yaml before modifications
# =================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="config.yaml"
BASE_EXPERIMENT_NAME=""
MODELS_TO_TRAIN=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name|-n)
            BASE_EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            MODELS_TO_TRAIN+=("$1")
            shift
            ;;
    esac
done

# If no models specified, train all
if [ ${#MODELS_TO_TRAIN[@]} -eq 0 ]; then
    MODELS_TO_TRAIN=("LSTM" "GRU" "CNN-LSTM" "CNN-GRU" "TCN")
    echo -e "${YELLOW}No models specified, training all: ${MODELS_TO_TRAIN[*]}${NC}"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file '$CONFIG_FILE' not found${NC}"
    exit 1
fi

# Backup original config
BACKUP_CONFIG="${CONFIG_FILE}.backup"
cp "$CONFIG_FILE" "$BACKUP_CONFIG"
echo -e "${GREEN}✓ Backed up config to ${BACKUP_CONFIG}${NC}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Results array
declare -A RESULTS_VAL
declare -A RESULTS_TEST
declare -A RESULTS_TIME

echo ""
echo "========================================================================"
echo "                    BATCH MODEL TRAINING"
echo "========================================================================"
echo "Config file: $CONFIG_FILE"
echo "Models to train (${#MODELS_TO_TRAIN[@]}): ${MODELS_TO_TRAIN[*]}"
echo "========================================================================"
echo ""

# Train each model
TOTAL_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!MODELS_TO_TRAIN[@]}"; do
    MODEL="${MODELS_TO_TRAIN[$i]}"
    MODEL_NUM=$((i + 1))
    TOTAL_MODELS=${#MODELS_TO_TRAIN[@]}
    
    echo ""
    echo "========================================================================"
    echo "[$MODEL_NUM/$TOTAL_MODELS] Training: $MODEL"
    echo "========================================================================"
    
    # Set experiment name
    if [ -n "$BASE_EXPERIMENT_NAME" ]; then
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}/${MODEL}"
    else
        EXPERIMENT_NAME="$MODEL"
    fi
    
    # Update config.yaml with sed
    # Use '|' as delimiter to avoid conflicts with '/' in EXPERIMENT_NAME
    sed -i "s|^experiment_name:.*|experiment_name: \"$EXPERIMENT_NAME\"|" "$CONFIG_FILE"
    sed -i "s|^model:.*|model: \"$MODEL\"|" "$CONFIG_FILE"
    
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Model: $MODEL"
    echo ""
    
    # Start training
    START_TIME=$(date +%s)
    
    if python3 main.py; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        
        # Extract results from log file
        EVAL_LOG="./results/$EXPERIMENT_NAME/logs/eval.txt"
        
        if [ -f "$EVAL_LOG" ]; then
            VAL_RMSE=$(grep "Average Validation Cells RMSE:" "$EVAL_LOG" | awk '{print $5}')
            TEST_RMSE=$(grep "Average Test Cells RMSE:" "$EVAL_LOG" | awk '{print $5}')
            
            RESULTS_VAL["$MODEL"]="${VAL_RMSE:-N/A}"
            RESULTS_TEST["$MODEL"]="${TEST_RMSE:-N/A}"
            RESULTS_TIME["$MODEL"]="${ELAPSED}s"
            
            echo -e "${GREEN}✓ $MODEL completed successfully${NC}"
            echo "  Val RMSE: ${VAL_RMSE:-N/A} Ah"
            echo "  Test RMSE: ${TEST_RMSE:-N/A} Ah"
            echo "  Time: ${ELAPSED}s"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo -e "${YELLOW}$MODEL completed but no eval log found${NC}"
            RESULTS_VAL["$MODEL"]="N/A"
            RESULTS_TEST["$MODEL"]="N/A"
            RESULTS_TIME["$MODEL"]="${ELAPSED}s"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    else
        echo -e "${RED}✗ $MODEL training failed${NC}"
        RESULTS_VAL["$MODEL"]="FAILED"
        RESULTS_TEST["$MODEL"]="FAILED"
        RESULTS_TIME["$MODEL"]="N/A"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

# Restore original config
cp "$BACKUP_CONFIG" "$CONFIG_FILE"
echo ""
echo -e "${GREEN}✓ Restored original config${NC}"

# Print summary table
echo ""
echo "========================================================================"
echo "                      RESULTS SUMMARY"
echo "========================================================================"
echo ""
printf "%-15s | %-12s | %-12s | %-12s\n" "Model" "Val RMSE" "Test RMSE" "Time"
echo "------------------------------------------------------------------------"

for MODEL in "${MODELS_TO_TRAIN[@]}"; do
    printf "%-15s | %-12s | %-12s | %-12s\n" \
        "$MODEL" \
        "${RESULTS_VAL[$MODEL]}" \
        "${RESULTS_TEST[$MODEL]}" \
        "${RESULTS_TIME[$MODEL]}"
done

echo "========================================================================"
echo ""
echo "Training Statistics:"
echo "  Successfully trained: $SUCCESS_COUNT/${#MODELS_TO_TRAIN[@]}"
echo "  Failed: $FAIL_COUNT/${#MODELS_TO_TRAIN[@]}"
echo "  Total time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo ""

# Find best model
BEST_MODEL=""
BEST_RMSE=999999
for MODEL in "${MODELS_TO_TRAIN[@]}"; do
    RMSE="${RESULTS_TEST[$MODEL]}"
    if [[ "$RMSE" =~ ^[0-9.]+$ ]]; then
        if (( $(echo "$RMSE < $BEST_RMSE" | bc -l) )); then
            BEST_RMSE="$RMSE"
            BEST_MODEL="$MODEL"
        fi
    fi
done

if [ -n "$BEST_MODEL" ]; then
    echo -e "${GREEN}BEST MODEL: $BEST_MODEL${NC}"
    echo "   Test RMSE: ${BEST_RMSE} Ah"
    echo "   Val RMSE: ${RESULTS_VAL[$BEST_MODEL]} Ah"
    if [ -n "$BASE_EXPERIMENT_NAME" ]; then
        echo "   Location: ./results/${BASE_EXPERIMENT_NAME}/${BEST_MODEL}/"
    else
        echo "   Location: ./results/${BEST_MODEL}/"
    fi
fi

echo ""
echo "========================================================================"
echo "All results saved to: ./results/"
echo "Config backup: $BACKUP_CONFIG"
echo "========================================================================"
