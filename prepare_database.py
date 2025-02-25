import pandas as pd
import os
from sentence_transformers import SentenceTransformer

## THIS CODE IS FOR PRECOMPUTING EMBEDDINGS FOR THE TOOLS_TOTAL FILE ##
## IT ONLY HAS TO BE RUN ONCE, AND THE RESULTING FILE CAN BE USED IN THE PIPELINE ## 

def main(input_file):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Step 1: Initialize the SentenceTransformer model (IMPORTANT: This model must be the same as the one used to encode human input in pipeline.py)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 2: Read in the file (path relative to the script directory)
    input_path = os.path.join(script_dir, "softwaregini_files", input_file)
    tools = pd.read_csv(input_path, sep=';')

    # Step 3: Compute embeddings for the relevant column "description"
    # Here we use the .encode() method of the SentenceTransformer model
    def compute_embeddings(description):
        if pd.isna(description) or description == "":  # Skip empty values
            return None
        return model.encode(description).tolist()  # Convert the vector to a list for CSV storage

    print("Calculating embeddings for the 'description' column...")
    tools['description_embeddings'] = tools['description'].apply(compute_embeddings)

    # Step 4: Save the file with precomputed embeddings in the script's directory
    # CSV does not support direct lists, so we save the embeddings as a Pickle
    output_path = os.path.join(script_dir, "oos_tools_total_with_embeddings.pkl")
    tools.to_pickle(output_path)

    print("The file has been successfully saved with embeddings.")

if __name__ == '__main__':
    main(input_file = "out_of_sample/tools.csv")