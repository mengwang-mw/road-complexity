# Array of files
files=(
  "selected_features_one_hot_complexity_16_continuous_demanding_all_crash.csv"
  "selected_features_one_hot_complexity_16_continuous_demanding_oneformerDriving_crash.csv"
  "selected_features_one_hot_complexity_16_continuous_demanding_oneformer_crash.csv"
#   "selected_features_one_hot_complexity_32_categorical_demanding_all_crash.csv"
#   "selected_features_one_hot_complexity_32_categorical_demanding_oneformerDriving_crash.csv"
#   "selected_features_one_hot_complexity_32_categorical_demanding_oneformer_crash.csv"
)

# Array of features
features=(
  "base_comp"
  "comp"
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
      log_file="log/${file_name}_${feature}_lr${lr//.}.log"
      
      # Execute the Python script
      echo "Running for file: $file, feature: $feature, lr: $lr"
      python modeling_classification_nn.py \
      --file "$file" --feature "$feature" --lr "$lr" > "$log_file"
    done
  done
done