#!/bin/bash
# filepath: /mlf11-shared/multimodal/chartqapro/run_multiple_trials.sh

set -e  # Exit on error

# MODEL="gemma-3-4b-it" 
# STRATEGY="direct"
# BASE_OUTPUT_DIR="/mlf11-shared/multimodal/chartqapro/results/gemma3-4b+CePO+Step0/direct"
# TRIALS=5
# OPENAI_API_KEY="serving-on-vllm-1"

MODEL=$1 
STRATEGY=$2
BASE_OUTPUT_DIR=$3
OPENAI_API_KEY=$4
API_BASE=$5  #http://localhost:8010/v1
QUESTION_TYPE=$6
USE_CEPO=$7
USE_GRID=$8
TRIALS_STR=$9
TRIALS=$((TRIALS_STR + 0)) # convert to integer

echo "MODEL: $MODEL"
echo "STRATEGY: $STRATEGY"
echo "BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
echo "OPENAI_API_KEY: $OPENAI_API_KEY"
echo "API_BASE: $API_BASE"
echo "QUESTION_TYPE: $QUESTION_TYPE"
echo "TRIALS: $TRIALS"
echo "USE_GRID: $USE_GRID"

echo "Running $TRIALS trials with model $MODEL using $STRATEGY strategy"

mkdir -p $BASE_OUTPUT_DIR

for i in $(seq 1 $TRIALS); do
    echo "📊 Running trial $i of $TRIALS"
    # Create a unique output directory for this trial
    TRIAL_OUTPUT_DIR="$BASE_OUTPUT_DIR/trial_$i"
    mkdir -p $TRIAL_OUTPUT_DIR

    if [ "$USE_CEPO" = "use_cepo" ]; then
        echo "PASSING CEPO --use_cepo arg"
        if [ "$USE_GRID" = "draw_grid_on_image" ]; then
            echo "PASSING draw_grid_on_image --draw_grid_on_image arg"
            python /mlf11-shared/multimodal/aarti/cb_ChartQAPro/generate_responses_qapro_parallel.py \
                --model $MODEL \
                --strategy $STRATEGY \
                --output_dir $TRIAL_OUTPUT_DIR \
                --api_base $API_BASE \
                --api_key $OPENAI_API_KEY \
                --only_question_types "$QUESTION_TYPE" \
                --use_cepo \
                --draw_grid_on_image
        else
            echo "NOT PASSING draw_grid_on_image --draw_grid_on_image arg"
            python /mlf11-shared/multimodal/aarti/cb_ChartQAPro/generate_responses_qapro_parallel.py \
                --model $MODEL \
                --strategy $STRATEGY \
                --output_dir $TRIAL_OUTPUT_DIR \
                --api_base $API_BASE \
                --api_key $OPENAI_API_KEY \
                --only_question_types "$QUESTION_TYPE" \
                --use_cepo

        fi
    else
        echo "NO --use_cepo arg"
        if [ "$USE_GRID" = "draw_grid_on_image" ]; then
            echo "PASSING draw_grid_on_image --draw_grid_on_image arg"
            python /mlf11-shared/multimodal/aarti/cb_ChartQAPro/generate_responses_qapro_parallel.py \
                --model $MODEL \
                --strategy $STRATEGY \
                --output_dir $TRIAL_OUTPUT_DIR \
                --api_base $API_BASE \
                --api_key $OPENAI_API_KEY \
                --only_question_types "$QUESTION_TYPE" \
                --draw_grid_on_image
        else
            echo "NOT PASSING draw_grid_on_image --draw_grid_on_image arg"
            python /mlf11-shared/multimodal/aarti/cb_ChartQAPro/generate_responses_qapro_parallel.py \
                --model $MODEL \
                --strategy $STRATEGY \
                --output_dir $TRIAL_OUTPUT_DIR \
                --api_base $API_BASE \
                --api_key $OPENAI_API_KEY \
                --only_question_types "$QUESTION_TYPE" 

        fi
    fi
    echo "✅ Completed trial $i"
done

echo "🔍 All trials complete. Now analyzing results..."

python /mlf11-shared/multimodal/aarti/cb_ChartQAPro/analyze_multiple_trials.py --base_dir $BASE_OUTPUT_DIR --model $MODEL --strategy $STRATEGY --trials $TRIALS

echo "Analysis complete!"