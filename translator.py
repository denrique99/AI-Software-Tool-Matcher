import asyncio
import aiohttp
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"

async def translate_text_async(session, text):
    """
    Sends a single text translation request asynchronously.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "temperature": 0.3,
        "top_p": 1,
        "messages": [
            {"role": "system", "content": "Translate the following text into English. If it is all already in English, reply exactly with the input i gave you. If no description is available then dont translate, ignore it and move on"},
            {"role": "user", "content": text}
        ]
    }

    async with session.post(API_URL, json=payload, headers=headers) as response:
        res = await response.json()
        return res.get("choices", [{}])[0].get("message", {}).get("content", text).strip()

async def translate_with_progress(session, text, pbar):
    result = await translate_text_async(session, text)
    pbar.update(1)
    return result


async def translate_column_to_english(input_file, column_to_translate='description'):
    if not os.path.isfile(input_file):
        print(f"ERROR: File '{input_file}' not found. Check the directory!")
        return None

    # Load DataFrame from CSV or use the provided DataFrame
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    else:
        df = pd.read_csv(input_file, sep=';')

    df[column_to_translate] = df[column_to_translate].fillna('')
    texts = df[column_to_translate].tolist()

    print(f"Translating {len(texts)} texts asynchronously...")

    async with aiohttp.ClientSession() as session:
        # Create a progress bar with total equal to number of texts
        with tqdm(total=len(texts)) as pbar:
            # Wrap each translation task to update the progress bar when done
            tasks = [translate_with_progress(session, text, pbar) for text in texts]
            translated_texts = await asyncio.gather(*tasks)

    df[column_to_translate] = translated_texts

    print("Translation complete.")
    return df


"""async def main():
    print("📂 Current Directory:", os.getcwd())  # Show working directory
    print("📜 Files in Directory:", os.listdir())  

    # Load CSV before translation
    df = pd.read_csv('softwaregini_files/organization_tools_2.csv', delimiter=';') 

    # Translate column
    translated_df = await translate_column_to_english(input_file='softwaregini_files/organization_tools_2.csv')

    # Ensure translation was successful before saving
    if translated_df is not None:
        output_file = 'softwaregini_files/translated_organization_tools.csv'
        translated_df.to_csv(output_file, index=False, sep=';')
        print(f"✅ Translated data saved to {output_file}")
    else:
        print("❌ Translation failed. No file was saved.")"""

"""# Run the async function in an event loop
asyncio.run(main())
"""