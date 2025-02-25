import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
from IPython.display import display
import Levenshtein

import os


def load_and_match(human_file, tools_with_embeddings_file = 'data/tools_total_with_embeddings.pkl', name_weight=0.55, similarity_cutoff=70.0, name_scorer=fuzz.token_set_ratio, model_type = 1):
    """
    Matches human descriptions and names with tools based on semantic similarity and fuzzy matching.

    Parameters:
        human_file (str): Path to the human dataset file.
        tools_with_embeddings_file (str): Path to the tools dataset with precomputed embeddings.
        name_weight (float): Weight for name similarity in the combined score.
        similarity_cutoff (float): Minimum similarity score (threshold) to retain a match.
        name_scorer (function): Function to compute name similarity.
        model_type (int): 1 for 'all-MiniLM-L6-v2' (fast and efficient), 2 for 'sentence-transformers/all-mpnet-base-v2' (slow but potentially higher quality).

    Returns:
        pd.DataFrame: DataFrame of matches sorted by combined score.
    """
    # Checks if human_file is a csv file or a DataFrame
    if isinstance(human_file, pd.DataFrame):
        human_df = human_file
    else:
        # Load human dataset
        human_df = pd.read_csv(human_file, delimiter=';')

    # Load tools dataset with precomputed embeddings
    tools_df = pd.read_pickle(tools_with_embeddings_file)

    # Initialize SentenceTransformer model
    if model_type == 1:
        model = SentenceTransformer('all-MiniLM-L6-v2') # fast and efficient
    elif model_type == 2:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # slow but higher quality

    # Step 1: Precompute embeddings for human descriptions (batch processing)
    print("Computing embeddings for human descriptions...")
    human_df['description'] = human_df['description'].fillna('')  # Replace NaN with empty string
    human_descriptions = human_df['description'].tolist()
    human_embeddings = model.encode(human_descriptions, batch_size=64)  # Batch encoding
    human_df['embedding'] = list(human_embeddings)

    # Step 2: Prepare tools embeddings and names by loading them from .pkl file
    tools_embeddings = tools_df['description_embeddings'].tolist()
    tools_descriptions = tools_df['description'].tolist()
    tools_names = tools_df['name'].tolist()
    tools_ids = tools_df['id'].tolist()  # ✅ Load tool_id


    # Step 3: Perform matching
    matches = []
    potential_matches = []
    print("Matching human descriptions to tools...")
    i = 0
    x = 0
    for idx, human_row in human_df.iterrows():
        human_name = human_row['name']
        human_description = human_row['description']
        human_embedding = human_row['embedding']

        # Compute cosine similarity between the human embedding and all tools embeddings
        description_similarities = np.round(cosine_similarity([human_embedding], tools_embeddings)[0], 2)
        # Add matches for the current human description
        for tool_idx, description_similarity in enumerate(description_similarities):
            tool_name = tools_names[tool_idx]
            tool_description = tools_descriptions[tool_idx]

            # Name similarity using rapidfuzz
            name_similarity = name_scorer(human_name, tool_name)
            
            # For tools with identical names and no descriptions, consider them a match.
            if  Levenshtein.distance(human_name.lower(), tool_name.lower()) == 0 and human_description == '':
                combined_score = 100
            # Combined score
            else: 
                combined_score = ((1-name_weight) * (description_similarity * 100) + name_weight * name_similarity )

            # Filter matches based on a similarity cutoff threshold
            if combined_score >= similarity_cutoff:
                i = i + 1
                matches.append({
                    'human_name': human_name,
                    'human_description': human_description,
                    'tool_name': tool_name,
                    'tool_description': tool_description,
                    'tool_id': tools_ids[tool_idx],
                    'description_similarity': description_similarity * 100,
                    'name_similarity': name_similarity,
                    'combined_score': combined_score

                })
            # Potential matches conditions:
            # Add matches that are in the top 50% range below the similarity cuttoff threshold
            if combined_score > similarity_cutoff * 0.50 and combined_score < similarity_cutoff:
                x = x + 1
                potential_matches.append({
                    'human_name': human_name,
                    'human_description': human_description,
                    'tool_name': tool_name,
                    'tool_description': tool_description,
                    'tool_id': tools_ids[tool_idx],
                    'description_similarity': description_similarity * 100,
                    'name_similarity': name_similarity,
                    'combined_score': combined_score
                    
                })


    print(f"Amount of total Matches: ", i)
    print(f"Amount of potential Matches: ", x)


    # Step 4: Create a DataFrame of matches
    matches_df = pd.DataFrame(matches)
    potential_matches = pd.DataFrame(potential_matches)

    # Sort matches by combined score in descending order
    matches_df = matches_df.sort_values(by='combined_score', ascending=False)
    potential_matches = potential_matches.sort_values(by='combined_score', ascending=False)

    # Retain the best match for each human_name
    initial_count = len(matches_df)

    # Keeps the highest scored match, drops duplicates based on the human_name column
    best_matches = matches_df.drop_duplicates(subset=['human_name'], keep='first')

    # Extracts the tool_ids list of the best matches
    best_matches_tool_ids = list(best_matches['tool_id'])

    # Keeps only the best scored duplicate match
    potential_matches = potential_matches.drop_duplicates(subset=['human_name'], keep='first')
    # Collects the dropped duplicates from the best matches and adds them to potential_matches
    additional_matches = matches_df[~matches_df.index.isin(best_matches.index)]
    potential_matches = pd.concat([potential_matches, additional_matches], axis=0)

    final_count = len(best_matches)
    potential_count = len(potential_matches)
    dropped_count = initial_count - final_count
    print(f"Number of dropped/deleted matches: {dropped_count}")
    print(f"Number of final matches: {final_count}")
    print(f"Number of potential matches: {potential_count}")

    return best_matches, potential_matches, best_matches_tool_ids


def evaluate_matching(final_matches_file, ground_truth_file, human_file):
    # Reads the final matches and ground truth files
    if isinstance(final_matches_file, pd.DataFrame):
        final_matches = final_matches_file
    else:
        final_matches = pd.read_csv(final_matches_file, delimiter=';')

    ground_truth = pd.read_csv(ground_truth_file, delimiter=';')

    # Checks if human_file is a csv file or a DataFrame
    if isinstance(human_file, pd.DataFrame):
        human_df = human_file
    else:
        # Load human dataset
        human_df = pd.read_csv(human_file, delimiter=';')
    
    # Lowercasing und Trimmen von Strings
    final_matches = final_matches.sort_values(by='combined_score', ascending=False).drop_duplicates(subset=['human_name'], keep='first')
    final_matches['human_name'] = final_matches['human_name'].str.strip().str.lower()
    final_matches['tool_name'] = final_matches['tool_name'].str.strip().str.lower()
    ground_truth['Human Label'] = ground_truth['Human Label'].str.strip().str.lower()
    ground_truth['Tool Label'] = ground_truth['Tool Label'].str.strip().str.lower()

    # Matches in ein einheitliches Format bringen
    final_set = set(tuple(x) for x in final_matches[['human_name', 'tool_name']].values)
    truth_set = set(tuple(x) for x in ground_truth[['Human Label', 'Tool Label']].values)
    
    # True/False Positives/Negatives berechnen
    true_positives = len(final_set.intersection(truth_set))
    false_positives = len(final_set - truth_set)
    false_negatives = len(truth_set - final_set)
    true_negatives = len(human_df) - (true_positives + false_negatives + false_positives)
    # Listen der False Positives und False Negatives
    false_positives_list = list(final_set - truth_set)
    false_negatives_list = list(truth_set - final_set)

    # Metriken berechnen
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Ergebnisse speichern
    results = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'True Positives': true_positives,
        'True Negatives': true_negatives,
        'False Positives': false_positives,
        'False Negatives': false_negatives,
      #  'False Positives List': false_positives_list,
      #  'False Negatives List': false_negatives_list
    }
    
    # Ergebnisse ausgeben
    for metric, value in results.items():
        if isinstance(value, list):
            print(f"{metric}:")
            # Uncomment the next line if you want to print the full lists
            # for item in value:
            #     print(item)
        else:
            print(f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}")

    return results


def lowercase_column(df, column):
# Function to lowercase strings in a pandas DataFrame column
    df[column] = df[column].astype(str).str.lower().replace('nan', '')
    return df


def remove_specialcharacters(df, column):
    # Function to remove all special characters except for '+', '-', '!', and German umlauts (ä, ü, ö)
    df[column] = df[column].astype(str).str.replace(r'[^a-zA-Z0-9äüöÄÜÖß+\-! ]', '', regex=True)
    df[column] = df[column].astype(str).str.replace(r'[()]', '', regex=True)
    return df

def remove_quotes(df, column):
    # Function to clean quotes from strings in a pandas DataFrame column
    df[column] = df[column].astype(str).str.replace('"', '', regex=False)
    return df

def clean_file(input_file):
    # Read the CSV file
    # Checks if human_file is a csv file or a DataFrame
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    else:
        # Load human dataset
        df = pd.read_csv(input_file, delimiter=';')

    print(f"Cleaning file...")

    # Process string columns
    for column in df.columns:
        if df[column].dtype == 'object':
            df = lowercase_column(df, column)
            df = remove_quotes(df, column)
    
    # Remove special characters from 'description' column
    if 'description' in df.columns:
        df = remove_specialcharacters(df, 'description')

    return df

def save_to_csv(data, filename):
    # Save the data to a CSV file
    data.to_csv(filename, index=False, encoding='utf-8', sep=';')
    print(f"Data saved to {filename}.csv")

# Example usage
#human_file = 'organization_tools_en_clean.csv'  # Path to human dataset
#tools_with_embeddings_file = 'tools_total_with_embeddings.pkl'  # Path to tools dataset with embeddings

# Perform matching
#final_matches, potential_matches = load_and_match(human_file, tools_with_embeddings_file)

#print("Final Matches:")
#display(final_matches)
#print("Potential Matches:")
#display(potential_matches)

# Save results to a CSV
#final_matches.to_csv('data/final_match.csv', index=False, encoding='utf-8', sep=';')

# cleaning of tools
# # Print the current working directory
# print(f"Current working directory: {os.getcwd()}")

# # # Cleaning of Softwaregini db file
# # human_file = 'softwaregini_files\\out_of_sample\\tools.csv'  # Absolute path to human dataset

# # cleaned_df = clean_file(human_file)

# # if cleaned_df is not None:
# #     # Save results to a CSV
# #     cleaned_df.to_csv('softwaregini_files\\out_of_sample\\tools_clean.csv', index=False, encoding='utf-8', sep=';')