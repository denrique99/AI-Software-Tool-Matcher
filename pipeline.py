import os
import pandas as pd
import asyncio

# Import your helper functions
from translator import translate_column_to_english
from local_matching import load_and_match, evaluate_matching, clean_file, save_to_csv
from langchain_matcher_async import process_all_batches_async

async def matching_algorithm(input_file):
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Starting pipeline in script directory:", script_dir)

    # Ensure the script starts in the correct directory
    os.chdir(script_dir)


    # If input_file is not an absolute path, assume it's in the script directory
    if not os.path.isabs(input_file):
        input_file = os.path.join(script_dir, input_file)

    # Construct paths for data and softwaregini_files folders (subfolders in the script directory)
    data_dir = os.path.join(script_dir, "data")
    softwaregini_dir = os.path.join(script_dir, "softwaregini_files")

    # Ensure the directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(softwaregini_dir, exist_ok=True)

    # Define the file path for ground truth data (used for evaluation)
    ground_truth_file = os.path.join(data_dir, "ground_truth_2.csv")

    # Load and preprocess data asynchronously
    df_async = await translate_column_to_english(input_file)
    df_async = clean_file(df_async)

    # Perform local matching and initial evaluation
    df_matched_async, potential_matches_async, matches_tool_id = load_and_match(df_async, tools_with_embeddings_file = "oos_tools_total_with_embeddings.pkl", similarity_cutoff=75)
    #results_before_llm = evaluate_matching(df_matched_async, ground_truth_file, input_file)
    #print("Results of local matching algorithm for a cutoff-threshold of 75:", results_before_llm)

    # Prepare potential matches for LLM processing
    potential_matches_async.columns = potential_matches_async.columns.str.strip()
    potential_matches_async = potential_matches_async[['human_name', 'human_description']].rename(
        columns={'human_name': 'name', 'human_description': 'description'}
    )

    # Process potential matches asynchronously using the LLM-based function
    df_potential_async, df_recommended_async = await process_all_batches_async(potential_matches_async, 95, 50)

    # Combine the initial matches with the LLM-recommended matches and re-evaluate
    df_final_async = pd.concat([df_matched_async, df_potential_async], axis=0)
    # Remove duplicate rows based on both human_name and tool_name
    df_final_async = df_final_async.drop_duplicates(subset=['human_name', 'tool_name'], keep='first')

    # Remove entries in df_recommended_async that already exist in df_final_async
    df_recommended_async = df_recommended_async.merge(
        df_final_async[['human_name', 'tool_name']], 
        on=['human_name', 'tool_name'], 
        how='left', 
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    # Clean the resulting DataFrame
    df_final_async = clean_file(df_final_async)

    # Evaluate the final matches
    #results_after_llm = evaluate_matching(df_final_async, ground_truth_file, input_file)
    #print("Results after LLM:", results_after_llm)

    # Save the final and potential matches CSV files in the data folder
    save_to_csv(df_final_async, os.path.join(data_dir, "pipeline_final_matches.csv"))
    save_to_csv(df_recommended_async, os.path.join(data_dir, "pipeline_potential_matches.csv"))

    return df_final_async, df_recommended_async, matches_tool_id

if __name__ == '__main__':
    #asyncio.run(matching_algorithm(input_file=r"softwaregini_files\organization_tools_2.csv"))
    asyncio.run(matching_algorithm(input_file="softwaregini_files/out_of_sample/organization_tools.csv"))
