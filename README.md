# AutoWikiUpdate

Code and Data will be released soon!


## Installation
clone the repository and install the required dependencies:

```bash
git clone https://github.com/gangiswag/AutoWikiUpdate.git
pip install -r requirements.txt
```

## Dataset Setup



## WiNELL
WiNELL consists of 3 modules - criteria extraction, news search and wikipedia editing. Each of these modules need to be run sequentially

To run the criteria extraction for a paritcular wikipedia entity
```bash
cd agent
export PAGE_ID=307
export OUTPUT_DIR="outputs/$PAGE_ID"

mkdir -p "$OUTPUT_DIR/criteria"

export TIME_DELTA=14
export ENTITY_FILE="../data/entities.json"
export DATA_DIR="../data/wikipedia_data"
export CRITERIA_DIR="$OUTPUT_DIR/criteria"

python3 extract_wikipedia_criteria.py \
    --model gpt-4.1-2025-04-14 \
    --entity_file $ENTITY_FILE \
    --data_dir $DATA_DIR \
    --out_dir "$CRITERIA_DIR" \
    --page_id $PAGE_ID \
    --time_delta $TIME_DELTA \
    --time_delta_file $OUTPUT_DIR/time_delay.json \
```

After running criteria extraction now we can run the the new search agent for a the same wikipedia entity. We run the search agent for every criteria file we have extracted.
```bash
cd agent
export PAGE_ID=307
export OUTPUT_DIR="outputs/$PAGE_ID"

mkdir -p "$OUTPUT_DIR/news_updates"

export CRITERIA_DIR="$OUTPUT_DIR/criteria"
export NEWS_UPDATES_DIR="$OUTPUT_DIR/news_updates"
export NUM_ITERATION=1

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
```

Finally we can now run the wikipedia editor to incorporate the new information in the wikipedia article. The `use_trained_model` flag enables the fine-tuned editor to be used for editing wikipedia, we also support using a prompted LLM for editing - pass the corresponding model in `model` flag
```bash
cd agent
export WORLD_SIZE=2
export PAGE_ID=307
export OUTPUT_DIR="outputs/$PAGE_ID"

mkdir -p "$OUTPUT_DIR/news_updates"
mkdir -p "$OUTPUT_DIR/updated_content"

export CRITERIA_DIR="$OUTPUT_DIR/criteria"
export NEWS_UPDATES_DIR="$OUTPUT_DIR/news_updates"
export UPDATED_CONTENT_DIR="$OUTPUT_DIR/updated_content"


for file in "$CRITERIA_DIR"/*.json; do
    # Extract the date from the filename after the underscore and before .json
    base=$(basename "$file")
    date_part=$(echo "$base" | cut -d'_' -f2 | cut -d'.' -f1)
    # Convert the date to MM-DD-YYYY format
    formatted_date=$(date -d "$date_part" +"%m-%d-%Y")
    # Read the time_delta from the JSON file using date_part as key
    td=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/time_delay.json'))['$date_part'])")
    # Incorporate the news updates into the sections
    echo "Integrating content for file: $file"
    python3 integrate_content_gpt.py \
        --model gpt-4.1-2025-04-14 \ # if use_trained_model not set, this model will be used
        --trained_model_path $TRAINED_MODEL_PATH \ # fine-tuned editor model
        --news_data_file "$NEWS_UPDATES_DIR/updates_$base" \
        --out_dir "$UPDATED_CONTENT_DIR" \
        --use_trained_model  # uses the fine-tuned editor
```

Alternatively we also provide a script to run all the modules together `run.sh`. Feel free to modify the CONSTANTS in the script.

```
bash run.sh $PAGE_ID
```

## Baselines





## Evaluations




