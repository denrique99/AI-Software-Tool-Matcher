import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm  # Progress bar
import asyncio
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "tools-matcher"
BATCH_SIZE = 50
# MIN_SCORE = 90

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

async def process_batch(batch_rows, vectorstore, llm, min_score):
    """Processes a batch of tools asynchronously."""
    match_results = []
    recommended_results = []
    prompts = []
    
    # Prepare human texts for similarity search
    human_texts = []
    for _, row in batch_rows.iterrows():
        description = str(row['description']) if pd.notna(row['description']) else ""
        name = str(row['name']) if pd.notna(row['name']) else ""
        human_texts.append(f"Name: {name}\nDescription: {description}")
    
    # Perform similarity searches in parallel using threads
    similar_docs_list = await asyncio.gather(
        *[asyncio.to_thread(vectorstore.similarity_search, ht, k=1) for ht in human_texts]
    )
    
    # Prepare prompts
    for i, similar_docs in enumerate(similar_docs_list):
        row = batch_rows.iloc[i]
        name = str(row['name']) if pd.notna(row['name']) else ""
        description = str(row['description']) if pd.notna(row['description']) else ""
        for doc in similar_docs:
            formatted_prompt = prompt_template.format(
                human_tool=name, human_desc=description,
                matched_tool=doc.metadata['name'], matched_desc=doc.page_content
            )
            prompts.append({
                'prompt': formatted_prompt,
                'metadata': {'human_name': name, 'matched_tool': doc.metadata['name']}
            })
    
    # Execute LLM calls asynchronously
    responses = await asyncio.gather(*[llm.ainvoke(p['prompt']) for p in prompts])
    
    # Process responses
    for prompt_data, response in zip(prompts, responses):
        score = response.content
        try:
            score_value = float(score)
            if score_value >= min_score:
                match_results.append({
                    'human_name': prompt_data['metadata']['human_name'],
                    'tool_name': prompt_data['metadata']['matched_tool'],
                    'combined_score': score_value
                })
            elif score_value < min_score and score_value >= 90.0 :
                recommended_results.append({
                    'human_name': prompt_data['metadata']['human_name'],
                    'tool_name': prompt_data['metadata']['matched_tool'],
                    'combined_score': score_value
                })
        except ValueError:
            continue
    
    return match_results, recommended_results

async def process_all_batches_async(df_human, min_score, batch_size, llm=llm_default, vectorstore=vectorstore_default, batch_delay_ms=50):
    """Processes all batches asynchronously with optional delay."""
    print(f"Started matching process with OpenAI GPT of {len(df_human)} tools in batches of {batch_size}...")
    matched_results_list = []
    recommended_results_list = []

    tasks = []
    for batch_num, start in enumerate(range(0, len(df_human), batch_size)):
        batch = df_human.iloc[start:start+batch_size]
        delay = batch_num * (batch_delay_ms / 1000)  # Convert ms to seconds
        task = asyncio.create_task(process_batch_with_delay(batch, vectorstore, llm, min_score, delay))
        tasks.append(task)
    
    # Use tqdm to show progress
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        batch_matched, batch_recommended = await f
        matched_results_list.extend(batch_matched)
        recommended_results_list.extend(batch_recommended)
    
    matched_df = pd.DataFrame(matched_results_list).sort_values('combined_score', ascending=False)
    recommended_df = pd.DataFrame(recommended_results_list).sort_values('combined_score', ascending=False)
    return matched_df, recommended_df

async def process_batch_with_delay(batch, vectorstore, llm, min_score, delay):
    await asyncio.sleep(delay)
    return await process_batch(batch, vectorstore, llm, min_score)

"""async def main():
    # Load dataset
    df_human = pd.read_csv('../softwaregini_files/organization_tools.csv', delimiter=';')
    matched_df, recommended_df = await process_all_batches_async(df_human, MIN_SCORE, BATCH_SIZE)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    os.makedirs(output_dir, exist_ok=True)
    matched_df.to_csv(os.path.join(output_dir, "matching_results.csv"), index=False, sep=';')
    recommended_df.to_csv(os.path.join(output_dir, "recommended_results.csv"), index=False, sep=';')

if __name__ == "__main__":
    asyncio.run(main())"""