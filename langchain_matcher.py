import os
import pandas as pd
from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


# Configuration
PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "tools-matcher"
BATCH_SIZE = 50
MIN_SCORE = 90

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore_default = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Initialize LLM
llm_default = ChatOpenAI(temperature=0, model="gpt-4o", api_key=OPENAI_API_KEY)

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["human_tool", "human_desc", "matched_tool", "matched_desc"],
    template="""Compare these software tools and determine if they are the same product.

    Tool 1:
    Name: {human_tool}
    Description: {human_desc}

    Tool 2:
    Name: {matched_tool}
    Description: {matched_desc}

    Scoring rules:
    100: Exact same product (identical names or official variants)
    95: Same product with minor name differences (e.g., full name vs. short name)
    90: Same product in different editions/versions
    0: Different products, even if similar purpose or from same company

    DO NOT match:
    - Different products from same company
    - Free vs. Pro versions as different products
    - Similar tools with different core purposes
    - Platform vs. specific service variants

    Respond ONLY with a number between 0-100."""
)

def process_batch(batch_rows, vectorstore, llm, min_score):
    """Processes a batch of tools and finds their closest match."""
    match_results = []
    recommended_results = []
    prompts = []
    
    # Prepare prompts
    for _, row in batch_rows.iterrows():
        description = str(row['description']) if pd.notna(row['description']) else ""
        name = str(row['name']) if pd.notna(row['name']) else ""
        human_text = f"Name: {name}\nDescription: {description}"
        
        similar_docs = vectorstore.similarity_search(human_text, k=1)
        
        for doc in similar_docs:
            formatted_prompt = prompt_template.format(
                human_tool=name, human_desc=description,
                matched_tool=doc.metadata['name'], matched_desc=doc.page_content
            )
            prompts.append({
                'prompt': formatted_prompt,
                'metadata': {'human_name': name, 'matched_tool': doc.metadata['name']}
            })

    # Execute LLM calls in parallel
    def process_prompt(prompt):
        return llm.invoke(prompt).content

    with ThreadPoolExecutor() as executor:
        batch_texts = [p['prompt'] for p in prompts]
        all_responses = list(executor.map(process_prompt, batch_texts))
    
    # Process responses
    for prompt_data, score in zip(prompts, all_responses):
        try:
            score_value = float(score)
            if score_value >= min_score:
                match_results.append({
                    'human_name': prompt_data['metadata']['human_name'],
                    'tool_name': prompt_data['metadata']['matched_tool'],
                    'similarity_score': score_value
                })
            elif score_value >= min_score - 15:
                recommended_results.append({
                    'human_name': prompt_data['metadata']['human_name'],
                    'tool_name': prompt_data['metadata']['matched_tool'],
                    'similarity_score': score_value
                })
        except ValueError:
            continue
    
    return match_results, recommended_results

def process_all_batches(df_human, min_score, batch_size, llm = llm_default, vectorstore = vectorstore_default):
    """Processes all batches in the dataset.
    
    Returns two DataFrames:
      - matched_df: DataFrame of matched results sorted by 'similarity_score'
      - recommended_df: DataFrame of recommended results sorted by 'similarity_score'
    """
    matched_results_list = []
    recommended_results_list = []

    for i in tqdm(range(0, len(df_human), batch_size)):
        batch = df_human.iloc[i:i+batch_size]
        batch_matched, batch_recommended = process_batch(batch, vectorstore, llm, min_score)
        matched_results_list.extend(batch_matched)
        recommended_results_list.extend(batch_recommended)
    
    matched_df = pd.DataFrame(matched_results_list).sort_values('similarity_score', ascending=False)
    recommended_df = pd.DataFrame(recommended_results_list).sort_values('similarity_score', ascending=False)
 
    return matched_df, recommended_df

"""def main():
    # Main function to execute tool matching.
    # Load dataset
    df_human = pd.read_csv('../softwaregini_files/organization_tools.csv', delimiter=';')
    df_results = process_all_batches(df_human, MIN_SCORE, BATCH_SIZE)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(os.path.join(output_dir, "matching_results.csv"), index=False, sep=';')
    
if __name__ == "__main__":
    main()
"""