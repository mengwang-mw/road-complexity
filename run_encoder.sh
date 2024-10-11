# Array of features
features=(
    "all"
    "oneformer"
    "oneformerDriving"
)

# Loop over features
for feature in "${features[@]}"; do
    # Construct log filename dynamically
    log_file="log/encoder_mturk_${feature}.log"

    # Execute the Python script
    echo "Running for file: $file, feature: $feature, lr: $lr"
    python feature_extraction.py \
    --response mturk --feature "$feature" > "$log_file"
done