#cost estimation 

import langchain
from langchain.document_loaders import DataFrameLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import io
import os
import shutil
import requests
import time

import numpy as np
import pandas as pd

import openai
import tiktoken
import unidecode

import litstudy
from scholarly import scholarly
from googlesearch import search

from fuzzywuzzy import fuzz
import PyPDF2

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def autosearch_missing_abstracts(file_path: str, outpath: str, bias = ["pubmed", "researchgate"]):
    """"
    this function replaces short (<50) and missing abstracts from the provided csv through calls to google scholar google.
    for proper scientific work, abstracts should be gathered manually rather than with this function, because blind reliance on google might result in false matches between papers and abstract.
    abstract source is added to output dataframe/file though.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='latin1')
    use_google = False
    for i in range(len(df)):
        print(f"""\n{shorten(df.loc[i, "Author"], 20)}: {shorten(df.loc[i, "Title"], 20)}""")
        if len(str(df.loc[i, "Abstract Note"])) < 50 or pd.isna(df.loc[i, "Abstract Note"]):
            if not use_google:
                try:
                    new_ref = get_abstract_from_scholar(dict(df.loc[i,:]))
                except Exception as e:
                    print(e)
                    use_google = True
            if use_google:
                print("AI OPPOSE MESSAGE: hit google scholar rate limit; using google")
                new_ref = get_abstract_from_google(dict(df.loc[i,:]), bias = bias)
            print(f"""abstract length from {len(str(df.loc[i, "Abstract Note"]))} to {len(new_ref["Abstract Note"])}""")
            df.loc[i, "Abstract Note"] = new_ref["Abstract Note"]
            df.loc[i, "SOURCE"] = new_ref["SOURCE"]
    if outpath:
        df.to_csv(outpath)
    return df


def clean_string(input_string, min_word_length = 0, filepath = False):
    """
    docstring pending
    """
    if filepath:
        paths = input_string.split("/")
        if len(paths) > 0:
            input_string = paths[-1]
        else:
            input_string = paths[0]
    punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '–', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    cleaned_string = ''.join(char if char not in punctuation_chars else " " for char in input_string)
    cleaned_string = ' '.join([word for word in cleaned_string.split() if len(word) >= min_word_length])
    return cleaned_string


def delete_duplicated_elements(mylist: list) -> list:
    """
    deletes langchain docs (internal usage)
    """
    unique_els = []
    for el in mylist:
        if el not in unique_els:
            unique_els.append(el)
    return unique_els


def delete_duplicated_docs(mylist: list) -> list:
    """
    deletes langchain docs (internal usage)
    """
    unique_els = []
    for el in mylist:
        if el[0] not in [e[0] for e in unique_els]:
            unique_els.append(el)
    return unique_els


def get_abstract_from_google(ref, bias = ["pubmed", "researchgate"]):
    """
    docstring pending
    """
    short_author = f"""{clean_string(ref["Author"], min_word_length = 2)}""".split()[0]
    results = search(f"""{' '.join(bias)} {short_author} {ref["Title"]}""", num_results=1)
    time.sleep(np.random.randint(15,35))
    url = list(results)[0]
    ref["Abstract Note"] = extract_abstract(url)
    ref["SOURCE"] += f"--> google: {url}"
    return ref


def get_abstract_from_scholar(ref):
    """
    docstring pending
    """
    results = scholarly.search_single_pub(ref["Title"]) #search_pubs
    time.sleep(np.random.randint(15,35))
    try:
        first_result = results #next(results)
    except StopIteration as e:
        try:
            short_title = shorten(f"""{clean_string(ref["Title"], min_word_length = 2)}""", 25)
            short_author = f"""{clean_string(ref["Author"], min_word_length = 2)}""".split()[0]
            results = scholarly.search_single_pub(f"{short_author} {short_title}")
            first_result = results
        except:
            ref["SOURCE"] += f"--> NO RESULTS ON SCHOLAR"
            return ref
    try:
        url = first_result["pub_url"]
        title = first_result["bib"]["title"]
        year = first_result["bib"]["pub_year"]
        authors = first_result["bib"]["author"]
    except:
        ref["SOURCE"] += f"--> NO VALID URL ON SCHOLAR"
        return ref
    title_orig = clean_string(ref["Title"].lower(), min_word_length=2)
    title_search = clean_string(title.lower(), min_word_length=2)
    titles_match = fuzz.WRatio(title_orig, title_search) > 85
    authors_orig = clean_string(ref["Author"].lower(), min_word_length=2)
    authors_search = clean_string(' '.join([authors[i].lower() for i in range(len(authors))]), min_word_length=2)
    authors_match = fuzz.WRatio(authors_orig, authors_search) > 70
    if is_integer(ref["Publication Year"]) and is_integer(year):
        year_match = np.abs(int(year) - int(ref["Publication Year"])) <= 1
    else: 
        year_match = True
    print(f"scholar found: {all([titles_match, authors_match, year_match])}")
    if titles_match and authors_match and year_match:
        ref["Abstract Note"] = extract_abstract(url)
        ref["SOURCE"] += f"--> {url}"
    else:
        ref["SOURCE"] += f"--> SCHOLAR RESULT LOOKS LIKE MISMATCH"
    return ref


def get_relevant_docs(presented_claim: str, chroma_directory: str, max_documents = 8, search_width = 2) -> any:
    """
    conducts various similarity searches between the given claim and the literature in the vectorstore.
    returns relevant langchain documents.
    """
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
        print(f"\nNo relevant documents found in your literature corpus '{chroma_directory}'! Try adding papers.\n")
        for doc in relevant_docs:
            doc.page_content = "No documents were relevant for the claim. No comment can be generated."
            doc.metadata = ""
    return relevant_docs

    
def extract_abstract(url):
    """
    docstring pending
    """
    chrome_options = Options()
    # chrome_options.add_argument('--headless') 
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(10)
    if url.endswith(".pdf") or url.endswith(".PDF"):
        page_text = get_pdf_url_text(url)
    elif "pubmed.ncbi" in url:
        page_text = driver.find_element(By.ID, 'eng-abstract').text
    elif "link.springer" in url:
        parent_div = driver.find_element(By.CLASS_NAME, 'c-article-body')
        page_text = parent_div.find_element(By.CSS_SELECTOR, "section[data-title='Abstract']").text
    else:
        try:
            try:
                abstract_divs = driver.find_elements(By.CSS_SELECTOR, '[class*="abstract"]')
                page_text = [abstract_div.text for abstract_div in abstract_divs if len(abstract_div.text) > 50][0]
            except:
                abstract_divs = driver.find_element(By.CSS_SELECTOR, '[id*="abstract"]')
                page_text = [abstract_div.text for abstract_div in abstract_divs if len(abstract_div.text) > 50][0]
        except:
            try:
                page_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Abstract')]/following-sibling::*[1]").text
            except:
                page_text = ""
    driver.quit()
    return page_text


def extract_last_names(text: str) -> str:
    """extracts last names by assuming they are followed by a comma"""
    try:
        words = text.split(" ")
        edited_text = " ".join(name for name in words if name[-1] == ",")
        return edited_text
    except:
        return text
    

def extract_pdf_references(filepath: str) -> any:
    """
    docstring pending
    """
    loader = PyPDFLoader(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
    docs = loader.load_and_split(text_splitter=text_splitter)
    if len(docs) == 0:
        print("no readable text found in pdf")
        return []
    responses = []
    system_prompt = "You are an AI tool that extracts reference lists from papers."
    no_hits = 0
    found_refs = False
    for doc in reversed(docs):
        prompt = f"Extract author names, paper titles, and years from the following bibliography. Respond in this list format: [{{'year': '1992', 'authors': 'E. Wiltens, A. Stone', 'title': 'effects of mdma'}}, {{'year': '2022', 'authors': 'S. Wiesen', 'title': 'durability of iron in water'}}]. Exclude journal names. Return an empty list [] if there was no bibliography or only short in-text references without paper titles. TEXT: {doc.page_content}"
        response = perform_chat_completion(prompt= prompt, system_message=system_prompt, temperature=0)
        ref_list = references_string_as_list(response)
        if len(ref_list) > 0:
            found_refs = True
        else:
            if found_refs: #if there were refs found previously then we might have gone up to the beginning of the bibliography
                no_hits += 1
                if no_hits == 2:
                    break
            continue
        responses.extend(ref_list)
    try:
        responses = [ref for ref in responses if ref["title"] != "" and ref["authors"] != ""]
        responses.sort(key= lambda x: clean_string(x["authors"], min_word_length=2))
        new_key_names = {"title": "Title", "year": "Publication Year", "authors": "Author"}
        responses = [rename_dict_keys(ref, new_key_names) for ref in responses]
    except:
        print("likely the LLM misformatted the pdf references")
    return delete_duplicated_elements(responses)


def find_metadata_for_ref(ref: dict) -> dict:
    """
    docstring pending
    """
    docs = litsearch(ref["Title"])
    for doc in docs:
        if doc.authors is None or doc.publication_year is None or doc.title is None:
            print(f"skipping incomplete search result: {doc.publication_year} || {' '.join([doc.authors[i].name.lower() for i in range(len(doc.authors))])} || {doc.title}")
            continue
        if doc.authors  == 'N/A' or doc.publication_year == 'N/A' or doc.title == 'N/A' :
            print(f"AAAAAAAAAAA skipping incomplete search result: {doc.publication_year} || {doc.authors} || {doc.title}")
            continue
        # print(f"""{ref["Publication Year"]} ||| {shorten(ref["Author"], 15)} ||| {shorten(ref["Title"].lower(), 15)}""")
        # print(f"""{doc.publication_year} ||| {shorten(' '.join([doc.authors[i].name.lower() for i in range(len(doc.authors))]), 15)} ||| {shorten(doc.title.lower(), 10)}""")
        #title
        title_orig = clean_string(ref["Title"].lower(), min_word_length=2)
        title_search = clean_string(doc.title.lower(), min_word_length=2)
        titles_match = fuzz.WRatio(title_orig, title_search) > 85
        # print(f"titles: {fuzz.WRatio(title_orig, title_search)}/85")
        #authors
        authors_orig = clean_string(ref["Author"].lower(), min_word_length=2)
        authors_search = clean_string(' '.join([doc.authors[i].name.lower() for i in range(len(doc.authors))]), min_word_length=2)
        authors_match = fuzz.WRatio(authors_orig, authors_search) > 70
        # print(f"authors: {fuzz.WRatio(authors_orig, authors_search)}/70")
        #year
        if is_integer(ref["Publication Year"]):
            year_match = np.abs(int(doc.publication_year) - int(ref["Publication Year"])) <= 1
        else: 
            year_match = False
        # print(f"""year difference {np.abs(int(doc.publication_year) - int(ref["Publication Year"]))}""")
        #match
        # print(f"{year_match}, {authors_match}, {titles_match}")
        if year_match and authors_match and titles_match:
            ref = {"Author": ' '.join([doc.authors[i].name.lower() for i in range(len(doc.authors))]), "Title": doc.title, "Publication Year": doc.publication_year, "Abstract Note": doc.abstract, "References": [], "Citations": []}
            if doc.references:
                ref["References"] =  [d for d in doc.references]
            if doc.citations:
                ref["Citations"] = [d for d in doc.citations]
            return ref
    #no match
    ref["Abstract Note"] = ""
    ref["References"] = []
    ref["Citations"] = []
    return ref


def filter_docs(presented_claim: str, relevant_docs: list) -> list:
    """
    This function checks whether the docs returned from the literature search are truly relevant. (internal usage)
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


def format_texts_from_csv(file_path: str, abstract_column: str, author_column: str, title_column: str, year_column: str) -> pd.DataFrame:
    """
    columns expected in csv (with default zotero names):
    "Abstract Note": abstract of the paper
    "Publication Year": year of the publication
    "Author": assumes last names are followed by comma
    "Title": title of paper
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except:
        data = pd.read_csv(file_path, encoding='latin1')
    chosen_cols = [abstract_column, author_column, title_column, year_column]
    if not all([col in data.columns for col in chosen_cols]):
        raise ValueError(f"make sure that your csv has the chosen column headers: {chosen_cols}")
    indices = data[data[abstract_column].isna() | data[abstract_column].eq('')].index
    if len(indices > 0):
        print(f'careful there are empty cells in column "{abstract_column}" resulting in empty literature entries.\nROWS: {indices}')
    for col in chosen_cols:
        data[col] = data[col].astype(str)
    data[author_column] = data[author_column].apply(extract_last_names)
    data["file_content"] = "TITLE: '" + data[title_column] + "' ABSTRACT: '" + data[abstract_column] + "' SOURCE: " + "("  + data[author_column]  + " " + data[year_column].astype(str) + ")"
    data["reference"] = [shorten(author, 80) + " "  + str(yea) for author, yea in zip(data[author_column], data[year_column].astype(str) )]
    data["reference"] = data["reference"].apply(unidecode.unidecode)
    data["file_content"] = data["file_content"].apply(unidecode.unidecode)
    return data


def generate_adversarial_texts(presented_claim: str, relevant_docs: list,  summarize_results = True, ) -> tuple:
    """
    This function produces the adversarial claims through gpt-4 and under reference of the relevant documents.
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
    print(f"\nThe file {opposition_file} has been generated with a source-wise response to the claim.")
    return (responses, summary_response)


def generate_literature_csv_from_pdf(pdf_path: str, author_column: str, title_column: str, year_column: str, output_path = None, max_secondary = 20) -> pd.DataFrame:
    """
    docstring pending
    """
    print("\n1/3 extracting references from pdf\n")
    refs = extract_pdf_references(pdf_path)
    if len(refs) == 0:
        return None
    data = pd.DataFrame(refs)
    data = remove_duplicate_rows(data)
    print(f"  --> {len(data)} refs extracted from pdf")
    finds = []
    entry_counts = {}
    print("\n2/3 finding abstracts of extracted refs and related papers\n")
    for i in range(data.shape[0]):
        ref = {"Author": data[author_column][i], "Title": data[title_column][i], "Publication Year": data[year_column][i]}
        find = find_metadata_for_ref(ref)
        finds.append(find)
        secondary_references = [r.doi for r in find["References"]]
        secondary_citations = [r.doi for r in find["Citations"]]
        ref_candidates = secondary_references + secondary_citations
        for entry in ref_candidates:
            entry_counts[entry] = entry_counts.get(entry, 0) + 1
    data_enriched = pd.DataFrame(finds)
    data_enriched["SOURCE"] = pdf_path
    selected_columns = [author_column, title_column, year_column, "Abstract Note"]
    filtered_secondary = {key: value for key, value in entry_counts.items() if key is not None}
    max_secondary = np.min([max_secondary, len(filtered_secondary.keys())])
    print(f"  --> {len(filtered_secondary.keys())} secondary refs found. limiting abstract search to {max_secondary}")
    filtered_secondary = dict(sorted(filtered_secondary.items(), key=lambda item: item[1], reverse=True)[:max_secondary])
    second_order_finds = []
    print("\n3/3 searching for abstracts of secondary literature\n")
    for doi in filtered_secondary:
        try:
            second_order_find = {}
            t = litstudy.fetch_semanticscholar(doi)
            second_order_find["Title"] = t.title
            second_order_find["Author"] = ' '.join([t.authors[i].name.lower() for i in range(len(t.authors))])
            second_order_find["Publication Year"] = t.publication_year
            second_order_find["Abstract Note"] = t.abstract
            second_order_find["SOURCE"] = doi
            second_order_finds.append(second_order_find)
        except:
            print(f"could not retrieve doi: {doi}")
    data_secondary_refs = pd.DataFrame(second_order_finds)
    combined_df = pd.concat([data_enriched[selected_columns + ["SOURCE"]], data_secondary_refs], ignore_index=True)
    combined_df['Abstract Note'] = combined_df['Abstract Note'].fillna('')
    combined_df['Abstract Note Length'] = combined_df['Abstract Note'].str.len()
    combined_df = combined_df.sort_values(by='Abstract Note Length', ascending=True)
    combined_df = combined_df.drop(columns=['Abstract Note Length'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df.to_csv(f"duplicates_{output_path}")
    combined_df_clean = remove_duplicate_rows(combined_df)
    if output_path:
        combined_df_clean.to_csv(output_path)
    print("\ndone.")
    return combined_df_clean


def generate_vectorstore(data: pd.DataFrame, filepath: str, max_doc_size = 4000) -> any:
    """generates a folder that should not be deleted as it will be used for the claim oppositions later. 
    it costs money to use this function as it is run with openai embedding model"""
    booleans = ["vectorstore_" in f for f in os.listdir()]
    if any(booleans):
        existing_vectorstore = [f for f, b in zip(os.listdir(),booleans) if b][0]
        userinput = input(f"Would you like to work with {existing_vectorstore} instead of generating a new vectorstore? (y/n)")
        if "y" in userinput.lower():
            return existing_vectorstore
    filepath = shorten(clean_string(filepath, remove_single_letters=False, filepath=True).replace("csv", ""), 40)
    new_chroma_directory = f'vectorstore_{filepath}_{time.ctime().replace(":", "").replace(" ", "")[0:-4]}'
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


def get_example_literature_file(lit = "egodepletion"):
    """
    possible values for lit argument are egodepletion, humorXdating, powerpose
    """
    if lit.lower() == "egodepletion":
        filename = "egodepletion_literature.csv"
    elif lit.lower() == "humorxdating":
        filename = "humorXdating_literature.csv"
    elif lit.lower() == "powerpose":
        filename = "powerpose_literature.csv"
    else:
        raise ValueError("lit argument should be one of 'egodepletion', 'humorXdating', 'powerpose'")
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    source_path = os.path.join(data_directory, filename)
    destination_path = filename
    print(f"collected file -> {destination_path}")
    shutil.copy(source_path, destination_path)


def get_pdf_url_text(pdf_url: str, max_len = 3500, split_on_abstract = True) -> str:
    """
    docstring pending
    """
    response = requests.get(pdf_url)
    pdf_data = response.content
    pdf_text = ''
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    pdf_text = pdf_text.lower().replace("\n","")
    if split_on_abstract:
        pdf_chunks = pdf_text.split("abstract")
        if len(pdf_chunks) > 1:
            pdf_text = pdf_chunks[1]
    if len(pdf_text) >= max_len:
        return pdf_text[:max_len]
    else:
        return pdf_text
    

def insert_linebreaks(text: str, interval: int) -> str:
    """
    string manipulation helper (internal usage)
    """
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


def is_integer(string):
    """
    docstring pending
    """
    try:
        int(string)
        return True
    except ValueError:
        return False


def litsearch(q: str) -> any:
    """
    docstring pending
    """
    q = clean_string(q)
    try:
        docs = litstudy.search_semanticscholar(q, limit = 5, batch_size=5)
    except:
        try:
            q_words = q.split(" ")
            q_shortened = " ".join(q_words[0:int(len(q_words)/2)])
            docs = litstudy.search_semanticscholar(q_shortened, limit = 4, batch_size=4)
        except:
            try:
                docs = litstudy.search_crossref(q, limit = 4)
            except:
                print(f"search fail for semantic scholar and crossref: {q}")
                docs = []
    docs = [doc for doc in docs]
    return docs


def perform_chat_completion(prompt: str, system_message: str, temperature=0) -> str:
    """
    openai wrapper (internal usage)
    """
    retries = 0
    max_retries = 5
    backoff_time = 5  # Initial backoff time in seconds
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": prompt}],
                temperature=temperature)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            retries += 1
            print(e)
            print(f"potential openai api issue (overload etc). trying again...({retries}/{max_retries})")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff
    return f"API Error {prompt}"


def references_string_as_list(input_string):
    """
    docstring pending
    """
    if not ("[" in input_string and "]" in input_string):
        print("no brackets found in extracted references")
        return []
    first_bracket_index = input_string.find('[')
    last_bracket_index = input_string.rfind(']')
    result_string = input_string[first_bracket_index:1+last_bracket_index]
    try:
        evaluated_list = list(eval(result_string))
        return evaluated_list
    except:
        print(f"Error: Could not evaluate the string as a list: {input_string} trimmed: {result_string}.")
        return []


def remove_duplicate_rows(df, similarity_threshold=90):
    """
    docstring pending
    """
    marked_duplicates = set()
    for outer_index, outer_row in df.iterrows():
        if outer_index in marked_duplicates:
            continue  # Skip rows that have already been marked as duplicates
        outer_author = outer_row['Author']
        outer_title = outer_row['Title']
        outer_year = outer_row['Publication Year']

        for inner_index, inner_row in df.iterrows():
            if outer_index != inner_index and inner_index not in marked_duplicates:
                inner_author = inner_row['Author']
                inner_title = inner_row['Title']
                inner_year = inner_row['Publication Year']
                author_similarity = fuzz.WRatio(outer_author, inner_author)
                title_similarity = fuzz.WRatio(outer_title, inner_title)
                if is_integer(outer_year) and is_integer(inner_year):
                    year_similarity = np.abs(int(inner_year) - int(outer_year))
                else:
                    year_similarity = 0
                if author_similarity > similarity_threshold and title_similarity > similarity_threshold and year_similarity <= 1:
                    marked_duplicates.add(inner_index)
    deduplicated_df = df.drop(index=marked_duplicates)
    return deduplicated_df


def rename_dict_keys(input_dict, key_mapping):
    """
    docstring pending
    """
    new_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in input_dict:
            new_dict[new_key] = input_dict[old_key]
    return new_dict


def shorten(text: str, length: int) -> str:
    """
    string manipulation helper (internal usage)
    """
    if len(text) > length:
        return text[0:length]
    return text