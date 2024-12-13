import os
import pandas as pd
from app import make_arules

# File and dataset paths

file_path = os.getenv('FILE_PATH', 'q_dir_motif_gene_shap_lag.csv')
output_file = 'allq_arules_df.csv'

# Check if the file already exists
if os.path.exists(output_file):
    print(f"[INFO] {output_file} already exists. Skipping generation.")
else:
    # Load dataset
    print("[INFO] Loading dataset...")
    data = pd.read_csv(file_path)

    # Generate association rules
    print("[INFO] Generating association rules...")
    support_threshold = 5  # Adjust threshold as needed
    score_threshold = 0
    allq_arules_df = make_arules(data, score_threshold=score_threshold)

    # Save to CSV
    allq_arules_df.to_csv(output_file, index=False)
    print(f"[INFO] Association rules saved to {output_file}")
