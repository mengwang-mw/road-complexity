# Array of files
files=(
  "selected_features_one_hot_complexity_32_continuous_demanding_all_crash_mturk.csv"
  "selected_features_one_hot_complexity_32_continuous_demanding_oneformerDriving_crash_mturk.csv"
  "selected_features_one_hot_complexity_32_continuous_demanding_oneformer_crash_mturk.csv"
)

# Array of features
features=(
#   "base"
  "base_comp"
  "comp"
  "base_index"
)

# Array of learning rates
learning_rates=(0.001)

# Loop over files
for file in "${files[@]}"; do
  # Extract part of the file name for log differentiation
  file_name=$(basename "$file" .csv) # Get the base file name without extension

  # Loop over features
  for feature in "${features[@]}"; do
    # Loop over learning rates
    for lr in "${learning_rates[@]}"; do
      # Construct log filename dynamically
      log_file="log/${file_name}_${feature}_lr${lr//.}_mturk.log"
      
      # Execute the Python script
      echo "Running for file: $file, feature: $feature, lr: $lr"
      python modeling_classification_nn.py \
      --file "$file" --feature "$feature" --lr "$lr" > "$log_file"
    done
  done
done