#!/bin/bash
set -e
set -o pipefail
source keys.env

# Get PAGE_ID from command line argument or prompt for it
if [ "$#" -ge 1 ]; then
    PAGE_ID="$1"
else
    read -p "Enter PAGE_ID: " PAGE_ID
fi

# CONSTANTS
export OUTPUT_DIR="final_outputs/$PAGE_ID"
# Configure for criteria extraction
export TIME_DELTA=14
export ENTITY_FILE="../data/entities.json"
export DATA_DIR="../data/wikipedia_data"
# Configure for news updates
export NUM_ITERATION=1
# Configure for content integration
export TRAINED_MODEL_PATH="gangiswag/llama3.1-8b-lora-wiki1000-0311-integrated"
# number of GPUs to use
export WORLD_SIZE=2

# define the output directory structure
mkdir -p "$OUTPUT_DIR/criteria"
mkdir -p "$OUTPUT_DIR/news_updates"
mkdir -p "$OUTPUT_DIR/updated_content"

export CRITERIA_DIR="$OUTPUT_DIR/criteria"
export NEWS_UPDATES_DIR="$OUTPUT_DIR/news_updates"
export UPDATED_CONTENT_DIR="$OUTPUT_DIR/updated_content"

# END

echo "Running criteria extractor agent for page ID: $PAGE_ID"
# Run the criteria extractor agent
python3 extract_wikipedia_criteria.py \
    --model gpt-4.1-2025-04-14 \
    --entity_file $ENTITY_FILE \
    --data_dir $DATA_DIR \
    --out_dir "$CRITERIA_DIR" \
    --page_id $PAGE_ID \
    --time_delta $TIME_DELTA \
    --time_delta_file $OUTPUT_DIR/time_delay.json \

# Run the news updates agent for each criteria file individually
for file in "$CRITERIA_DIR"/*.json; do
    # Extract the date from the filename after the underscore and before .json
    base=$(basename "$file")
    date_part=$(echo "$base" | cut -d'_' -f2 | cut -d'.' -f1)
    # Convert the date to MM-DD-YYYY format
    formatted_date=$(date -d "$date_part" +"%m-%d-%Y")
    # Read the time_delta from the JSON file using date_part as key
    td=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/time_delay.json'))['$date_part'])")
    # search websites for news updates
    echo "Running news updates agent for file: $file with date: $formatted_date and time delta: $td"
    python3 run_news_updates.py \
        --navigator_model gpt-4.1-mini \
        --aggregator_model gpt-4.1-mini \
        --extractor_model gpt-4.1-mini \
        --num_iterations $NUM_ITERATION \
        --wiki_file "$file" \
        --out_path "$NEWS_UPDATES_DIR/updates_$base" \
        --log_path "logs/log_$base.txt" \
        --start_time "$formatted_date" \
        --time_delta "$td" \

    # Incorporate the news updates into the sections
    echo "Integrating content for file: $file"
    python3 integrate_content_gpt.py \
        --model gpt-4.1-2025-04-14 \
        --trained_model_path $TRAINED_MODEL_PATH \
        --news_data_file "$NEWS_UPDATES_DIR/updates_$base" \
        --out_dir "$UPDATED_CONTENT_DIR" \
        --use_trained_model

done
