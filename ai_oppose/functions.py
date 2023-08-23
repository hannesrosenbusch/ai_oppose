import langchain
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import shutil
import openai
import pandas as pd
import time
import tiktoken
import unidecode

def get_egodepletion_example_lit():
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    source_path = os.path.join(data_directory, 'egodepletion_literature.csv')
    destination_path = 'egodepletion_literature.csv'
    shutil.copy(source_path, destination_path)

def insert_linebreaks(text: str, interval: int) -> str:
    lines = []
    current_line = ""
    for word in text.split():
        if word in ["ABSTRACT:", "SOURCE:"]:
            lines.append(current_line.strip())
            current_line = word + " "
        elif len(current_line) + len(word) <= interval:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return '\n'.join(lines)


def extract_last_names(text: str) -> str:
    """extracts last names by assuming they are followed by a comma"""
    try:
        words = text.split(" ")
        edited_text = " ".join(name for name in words if name[-1] == ",")
        return edited_text
    except:
        return text


def shorten(text: str, length: int) -> str:
    if len(text) > length:
        return text[0:length]
    return text


def delete_duplicated_docs(mylist: list) -> list:
    unique_els = []
    for el in mylist:
        if el[0] not in [e[0] for e in unique_els]:
            unique_els.append(el)
    return unique_els


def perform_chat_completion(prompt: str, system_message: str, temperature=0) -> str:
    retries = 0
    max_retries = 5
    backoff_time = 5  # Initial backoff time in seconds
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response["choices"][0]["message"]["content"].strip()
        except:
            print(f"potential openai api issue (overload etc). trying again...({retries}/{max_retries})")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff
            retries += 1
    return f"API Error {prompt}"


def format_texts_from_csv(file_path: str, abstract_column: str, author_column: str, title_column: str, year_column: str) -> pd.DataFrame:
    """
    columns expected in csv (zotero default):
    "Abstract Note": abstract of the paper
    "Publication Year": year of the publication
    "Author": assumes last names are followed by comma
    "Title": title of paper
    """
    now = time.ctime().replace(":", "").replace(" ", "")[0:-4]
    data = pd.read_csv(file_path)
    for col in [abstract_column, author_column, title_column, year_column]:
        data[col] = data[col].astype(str)
    data[author_column] = data[author_column].apply(extract_last_names)
    data["file_content"] = "TITLE: " + data[title_column] + " ABSTRACT: " + data[abstract_column] + " SOURCE: " + "("  + data[author_column]  + " " + data[year_column].astype(str) + ")"
    data["reference"] = [shorten(author, 80) + " "  + str(yea) for author, yea in zip(data[author_column], data[year_column].astype(str) )]
    data["reference"] = data["reference"].apply(unidecode.unidecode)
    data["file_content"] = data["file_content"].apply(unidecode.unidecode)
    return data


def generate_vectorstore(data: pd.DataFrame, max_doc_size = 4000) -> any:
    booleans = ["vectorstore_" in f for f in os.listdir()]
    if any(booleans):
        existing_vectorstore = [f for f, b in zip(os.listdir(),booleans) if b][0]
        userinput = input(f"Would you like to work with {existing_vectorstore} instead of generating a new vectorstore? (y/n)")
        if "y" in userinput.lower():
            return existing_vectorstore
    else:
        new_chroma_directory = f'vectorstore_{time.ctime().replace(":", "").replace(" ", "")[0:-4]}'
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
        loader = DataFrameLoader(data[["file_content","reference"]], page_content_column = "file_content")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_doc_size, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        if len(data) != len(data):
                print(f"Careful! {len(data)} files were saved as {len(docs)} documents in the vector store.")
        if os.path.exists(new_chroma_directory) and os.path.isdir(new_chroma_directory):
            if len(os.listdir(new_chroma_directory)) > 0:
                print("Careful! You already seem to have files in your new_chroma_directory. You might create duplicates which affect search results.")
        docsearch = Chroma.from_documents(docs, embeddings, persist_directory=new_chroma_directory)
        docsearch.persist()
        print(f"vectorstore has been generated: folder {new_chroma_directory}\n\n")
        return new_chroma_directory



def filter_docs(presented_claim: str, relevant_docs: list) -> list:
    """
    This function checks whether the docs returned from the literature search are truly relevant.
    """
    openai.api_key = os.environ['OPENAI_API_KEY']
    filtered_docs = []
    for doc, score in relevant_docs:
        filterprompt = f"""
        Presented claim:
        {presented_claim}

        Target paper:
        {doc.page_content}

        Instructions:
        Classify whether the Target Paper is relevant to the Presented Claim. Answer "yes" or "no". If unsure, answer "no".
        """
        filterresponse = perform_chat_completion(filterprompt, system_message="You are an attentive academic machine that only says 'yes' or 'no'.", temperature=0)

        if "yes" in filterresponse.lower():
            filtered_docs.append((doc, score))
    return filtered_docs


def get_relevant_docs(presented_claim: str, chroma_directory: str, max_documents = 8, search_width = 2) -> any:
    openai.api_key = os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

    vectordb = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)
    relevant_docs = []

    # #V1 basic claim opposition
    queryV1 = f"""The following claim is incorrect: "{presented_claim}" Other papers present opposing arguments."""
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV1, k=search_width))

    #V2 basic claim refinement
    queryV2 = f"""Someone wrote this oversimplification: "{presented_claim}" However, researchers disagree."""
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV2, k=search_width))

    #V3 pre-reversal of claim
    relevant_docs = delete_duplicated_docs(relevant_docs)
    queryV3 = perform_chat_completion(f'This is an academic claim: "{presented_claim}". Write a new claim that is the opposite of the original while using different words. Focus on the essence of the original.', "You are a good writer that is able to edit and revise text as instructed.", temperature=0)
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV3, k=len(relevant_docs) + 1))

    #V4 pre-summary of claim with opposition
    relevant_docs = delete_duplicated_docs(relevant_docs)
    responseV4 = perform_chat_completion(f'This is an academic claim: "{presented_claim}". Write a concise one-sentence summary of it using different words.', "You are a good writer that is able to edit and revise text as instructed.", temperature=0)
    queryV4 = f"Doubt surrounds this idea: {responseV4}"
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV4, k=len(relevant_docs) + 1))

    #V5 paraphrazation
    relevant_docs = delete_duplicated_docs(relevant_docs)
    responseV5 = perform_chat_completion(f'This is an academic claim: "{presented_claim}". Write a loose, concise paraphrazation.', "You are a good writer that is able to edit and revise text as instructed.", temperature=0)
    queryV5 = f"Doubt surrounds this idea: '{responseV5}'"
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV5, k=len(relevant_docs) + 1))

    #V6 meta-analyses and reviews
    queryV6 = f"meta-analyses and reviews have been written on this issue: {responseV5}"
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV6, k=len(relevant_docs) + 1))
    relevant_docs = delete_duplicated_docs(relevant_docs)
    relevant_docs = filter_docs(presented_claim, relevant_docs)
    if max_documents is not None:
        if not isinstance(max_documents, int):
            raise ValueError(f"max_documents must be int, but is {max_documents}")
        relevant_docs = sorted(relevant_docs, key=lambda x: x[1]) #prioritize high similarity results
        if max_documents < len(relevant_docs):
            relevant_docs = relevant_docs[0:max_documents]
    relevant_docs = [doc[0] for doc in relevant_docs]
    if len(relevant_docs) == 0:
        raise ValueError("No relevant documents found in AI vectorstore.")
    return relevant_docs



def generate_adversarial_texts(presented_claim: str, relevant_docs: list,  summarize_results = True, ) -> tuple:
    """
    REQUIRED
    This function produces the adversarial claims.
    """
    openai.api_key = os.environ['OPENAI_API_KEY']

    responses = []
    encoding = tiktoken.encoding_for_model("gpt-4")

    for doc in relevant_docs:
        prompt1 = f"""
        Presented claim:
        {presented_claim}

        Previous paper:
        {doc.page_content}

        Instructions:
        Refine, extend, or oppose the presented claim by means of the previous paper. Provide the apa reference in text. Use two concise sentences.
        """
        response = perform_chat_completion(prompt1, "You are a reviewer of an academic claim; you stick to the facts from the literature.", temperature=0)
        responses.append({"response": response, "metadata": doc.metadata, "page_content": doc.page_content})

    if summarize_results and len(responses) > 0:
        nr_tokens = 0
        all_comments = ""

        #counting tokens
        for response in responses:
            nr_tokens += len(encoding.encode(response["response"]))
            all_comments = f"""{all_comments}\nComment: {response["response"]}"""

        summary_prompt = f"""
        Presented claim:
        {presented_claim}

        Comments:
        {all_comments}

        Instructions:
        Summarize the comments to the claim into a coherent text. Keep all the relevant apa references. Keep it concise.
        """
        summary_response = perform_chat_completion(summary_prompt, "You are a reviewer of an academic claim; you stick to the facts from the literature.", temperature=0)

    else:
        summary_response = None
    opposition_file = f"""opposition_{time.ctime().replace(":", "").replace(" ", "")[0:-4]}.txt"""
    with open(opposition_file, "w", encoding="utf-8") as file:
        file.write(f"PRESENTED CLAIM: {presented_claim}\n\n")
        file.write(f"AI used {len(responses)} references.\n\n")
        for i, entry in enumerate(responses):
            file.write(f"{i+1})\n")
            file.write(f"""AI RESPONSE TO CLAIM:\n{insert_linebreaks(entry["response"], 140)}\n""")
            file.write(f"""SOURCE REF:\n{insert_linebreaks(entry["metadata"]["reference"], 200)}\n""")
            file.write(f"""SOURCE TEXT:\n{insert_linebreaks(entry["page_content"], 140)}\n\n\n""")
    print(f"\nThe file {opposition_file} has been generated with a source-wise response to the claim. One should work with the opposition file, but this is a very brief preview of some of the arguments:\n\n'{summary_response}'")
    return (responses, summary_response)