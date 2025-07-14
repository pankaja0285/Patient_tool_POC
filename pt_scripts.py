from dotenv import load_dotenv
import os 

# import google.generativeai as genai
from tqdm import tqdm
from tqdm.notebook import tqdm
# from google.generativeai import types
from IPython.display import Markdown, HTML, display
# from .autonotebook import tqdm as notebook_tqdm
import pprint
import time
import pandas as pd 
import numpy as np
import json5
import ast
import json
import re
from google import genai
from google.genai import types
import faiss
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# check version
genai.__version__



# ******************************************************************
# Scripts for set up and download data if need be
# ******************************************************************
def init_google_api_setup():
    # Load the .env file with the Google api key
    load_dotenv()

    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

def show_available_models():
    avl_models = genai.list_models()

    for model in avl_models:
        print(model.name)

def set_googleapi_model(default_model="gemini-2.0-flash-001"):
    # create a genai Generetive model
    t_model = genai.GenerativeModel(f"models/{default_model}") # "gemini-2.0-flash")
    pprint.pprint(t_model)
    return t_model

# Scripts for downloading data
def load_data(data_file=f"./data/medical_transcriptions_raw.csv"):
    df = pd.read_csv(data_file, index_col=None)
    df.head()
    return df


# ******************************************************************
# Scripts for prompt - content generation
# ******************************************************************
# clean and parse json
def clean_and_parse_json(text):
    # Remove triple backticks and language hints
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    loaded_json = {}

    try:
        loaded_json = json5.loads(cleaned)
    except Exception as e:
        print("⚠️ Parsing failed! Raw cleaned text:\n", cleaned)
        print("⚠️ Error:", e)
        
    return loaded_json

def generate_formatted_content(client, note: str, specialty: str = "general", model_name="") -> dict:
    # NOTE: prompt here is currently hard-coded, it can be extracted out into a config / yaml file
    #       or however one prefers
    prompt = f"""
    As a {specialty.lower()} doctor, extract structured data from the following patient transcripion.
     Return output ONLY as JSON with these keys:
    - Name
    - Age
    - Gender
    - Symptoms
    - Diagnoses
    - Medications
    - Tests

    DO NOT include ```json or any markdown.
    DO NOT explain anything.
    Patient Note:
    \"\"\"{note}\"\"\"

    Respond ONLY with valid JSON. Do not include triple backticks or explanation.
    """
    response = client.models.generate_content(model=model_name, contents=prompt)
    return clean_and_parse_json(response.text)


def safe_literal_eval(val):
    safe_eval = None
    try:
        if isinstance(val, str):
            safe_eval = ast.literal_eval(val)
        else:
            safe_eval = val
    except (ValueError, SyntaxError): # Catch both ValueError and SyntaxError
        print(f"ValueError Exception applying ast literal_eval") # Or handle as appropriate
    return safe_eval


def download_data_and_set_content(client=None, model_name="", data_file="", raw_file="", 
                                  update_to_existing=True, default_download=15):

    print(f"data_file: {data_file}")
    df_chunk = None
    df_existing = None
    df_concat = None
    step = ""

    try:
        # load the raw file
        step = " load data "
        df_raw = load_data(raw_file)

        if update_to_existing:
            # sample data --> data_file = f"./data/clinical_notes_sample.csv"
            df_existing = load_data(data_file)
            start = df_existing.shape[0]
            end = start + default_download
        else:
            start = 0
            end = default_download
        
        print(f"start, end: {start}, {end}")
        df_chunk = df_raw[start:end].copy(deep=True)
        print(df_chunk.head(2))

        step = " tqdm progress_apply "
        # Initialize tqdm for pandas
        tqdm.pandas()

        # here update only chunk as I am using free apikey
        df_chunk['structured_json'] = df_chunk.progress_apply(
            lambda row: generate_formatted_content(client,
                row['transcription'],  row['medical_specialty'],
                model_name=model_name),
            axis=1
        )

        # concat the 100 rows
        step = " concat "
        if update_to_existing:
            dfs = [df_existing, df_chunk]
            df_concat = pd.concat(dfs)
        else:
            df_concat = df_chunk
        print(f"Total concatenated rows: {df_concat.shape[0]}\n")
        
        # apply ast to clean up the json
        # ------------------------------------------------------
        # The 'structured_json' column may contain JSON-like strings.
        # This line ensures all such string entries are safely converted
        # into actual Python dictionaries using `ast.literal_eval()`.
        #
        # Why `ast.literal_eval`?
        # - Safer than `eval()` as it only parses literals (e.g., dicts, lists, strings, numbers).
        # - Essential for performing structured operations or key-based access later.
        #
        # If the entry is already a dictionary, it’s left unchanged.
        # create a structured_json column using ast.literal_eval
        
        # save the file back
        save_file = "./data/clinical_notes_sample.csv"
        df_concat.to_csv(save_file, index=None)
    except Exception as ex:
        print(f"Error occurred with exception: {ex} at step: {step}")
    return df_concat


# ***************************************************************************************
# Scripts for the bot/portal
# ***************************************************************************************

def embed_content(df, client=None, text_model_name="text-embedding-004"):
    try:
        texts = [text for text in df['rag_text']]

        # response = client.models.embed_content(
        #   model='text-embedding-004',
        #   contents=texts,
        # )

        response = client.models.embed_content(
                model=f'models/{text_model_name}',
                contents=texts,
                config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT')
            )
        # print(response.embeddings)

        if response:
            # apply the embeddings to create a new column
            df['embedding'] = [e.values for e in response.embeddings]
    except Exception as ex:
        print(f"Error occurred: {ex}")
    return df

def format_col_data_to_json(json_col_data):
    # Convert single-quoted string to a Python dictionary
    json_dict = ast.literal_eval(json_col_data)

    # Convert Python dictionary to a double-quoted JSON string
    double_quoted_json_str = json.dumps(json_dict)

    # Convert the double-quoted JSON string to a JSON object (Python dictionary)
    json_object = json.loads(double_quoted_json_str)

    return json_object


def create_rag_text(data):
    curr_rag_text = None

    try:
        # create rag_text
        def safe_list(val):
            return val if isinstance(val, list) else []

        curr_rag_text = (
            f"Patient Info:\n"
            f"Age: {data.get('Age', 'N/A')}\n"
            f"Gender: {data.get('Gender', 'N/A')}\n\n"
            f"Symptoms:\n" + '\n'.join(safe_list(data.get('Symptoms'))) + "\n\n"
            f"Diagnoses:\n" + '\n'.join(safe_list(data.get('Diagnoses'))) + "\n\n"
            f"Medications:\n" + '\n'.join(safe_list(data.get('Medications'))) + "\n\n"
            f"Tests:\n" + '\n'.join(safe_list(data.get('Tests')))
        )
    except Exception as ex:
        print(f"Error occurred as exception: {ex}")
    return curr_rag_text

def prep_rag_data(data_file="./data/clinical_notes_sample.csv", client=None, emb_max=75):
    step = ""
    df_data_with_rag = None

    try:
        # load the structured_json - data 
        # data_file = "./data/clinical_notes_sample.csv"
        step = " load data "
        df_data_with_rag_orig = load_data(data_file=data_file)
        
        step = " format_to_json "
        print(f"Applying format to json in prep_rag_data on loaded data...")

        # convert to structured_json to proper json object
        df_data_with_rag_orig['structured_json'] = df_data_with_rag_orig.apply(lambda x: format_col_data_to_json(x['structured_json']), axis=1)
        print(df_data_with_rag_orig.head(2))
        print(f"Applying format to json completed.")

        # create the rag_text column 
        step = " create rag_text "
        print("creating rag_text")
        # df['Existing_Column'].apply(lambda x: x * 2)
        df_data_with_rag_orig['rag_text'] = df_data_with_rag_orig['structured_json'].apply(lambda x: create_rag_text(x))
        print(df_data_with_rag_orig.head(2))
        print("creating rag_text completed")

        # create embedding
        step = " create embedding 1"
        print(step)
        # embed content for the rag_text
        # embed content 75 at a time as there's a 100 rows embedding limit   
        # hence commented below line
        # df_data_with_rag = embed_content(df_data_with_rag, client=client)
         
        dfs = []
        start = 0
        end = emb_max
        total_chunks = len(df_data_with_rag_orig) // emb_max

        for i in range(total_chunks):
            start = i * emb_max
            end = start + emb_max
            curr_df = df_data_with_rag_orig.iloc[start:end]
            curr_df = embed_content(curr_df, client=client)
            dfs.append(curr_df)

        # Add the remaining rows as the last chunk
        step = " create embedding 2"
        print(step)
        remaining_rows_start = total_chunks * emb_max
        if remaining_rows_start < len(df_data_with_rag_orig):
            curr_df = df_data_with_rag_orig.iloc[remaining_rows_start:]
            curr_df = embed_content(curr_df, client=client)
            dfs.append(curr_df)
        print("creating embedding completed")

        # concat the dfs
        step = " concatenate"
        df_data_with_rag = pd.concat(dfs)
        print(f"Prep rag data completed.")
        
    except Exception as ex:
        print(f"Error occurred as exception: {ex} at step: {step}")
    return df_data_with_rag

def create_index(df):
    # create the faiss index for the embeddings
    embedding_matrix = np.vstack(df['embedding'].values)
    dimension = embedding_matrix.shape[1]
    f_index = faiss.IndexFlatL2(dimension)
    f_index.add(embedding_matrix)
    print(f"Faiss index for embeddings created.")
    return f_index


def doctor_chatbot(question, df=None, 
                   client=None, genai_model_name="", 
                   text_model_name="text-embedding-004",
                   faiss_index=None, emb_max=75):
    result = None
    try:
        df_data_rag = None
        # check if the global dataframe variable is there or not
        df_data_rag = prep_rag_data(client=client)
        # create a copy
        df = df_data_rag.copy(deep=True)
        
        # check if index is alreaddy created, else create
        if faiss_index is None:
            print("Index not present, creating now...")
            faiss_index = create_index(df)

        # Step 1: Embed the question
        print("Creating query embedding...")
        query_embedding = client.models.embed_content(
            model=f'models/{text_model_name}',
            contents=question,
            config=types.EmbedContentConfig(task_type='retrieval_document')  # better for context fetching
        ).embeddings[0].values

        # Step 2: Search similar cases using FAISS
        print("Search for similar document with Faiss...")
        D, I = faiss_index.search(np.array([query_embedding]), k=3)
        top_docs = [df.iloc[i]['rag_text'] for i in I[0]]
        print(f"Total search documents found: {len(top_docs)}")

        # Step 3: Create the prompt
        context = "\n\n".join(top_docs)
        prompt = f"""
        You are a highly knowledgeable and cautious AI clinical assistant.
        Use the patient records below to provide a helpful response to the doctor's question.

        --- Patient Records (Context) ---
        {context}

        --- Doctor's Question ---
        {question}

        --- Instructions ---
        - Give your answer in markdown format.
        - Structure output clearly (e.g., **Possible Diagnoses**, **Treatment Suggestions**, **Next Steps**).
        - If the answer includes medical recommendations, always include a disclaimer.
        """

        # Step 4: Send prompt to Gemini
        print("Generate content response...")
        response = client.models.generate_content(
        model=genai_model_name,     #'gemini-2.0-flash',
        contents=prompt)
        print(f"Response generated: {response}")
        result = response.text
    except Exception as ex:
        print(f"Error occurred with exception: {ex}")
    return result     
