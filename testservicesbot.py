__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import chromadb.config
import fitz  
import os
import io
import json
import boto3
import random
import time
import warnings
import numpy as np
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

pdf_path = "POET_Everyday_Instructions_2page.pdf"
# keyword_image_map = create_keyword_mapping()

warnings.filterwarnings("ignore", category=FutureWarning)

bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id= os.environ['AWS_ACCESS_KEY'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

GREETINGS = "Hello! I am the Fedway Assistant. I can help you find product images. Please ask me about any product and I will display the images for you."

# Function to create a mapping between keywords and images
def create_keyword_mapping():
    keyword_image_map = {
        "Every Morning": ["extracted_images/page_1_img_1.png"],
        "Setting an active customer": ["extracted_images/page_1_img_2.png", "extracted_images/page_1_img_3.png"],
        "Sending an order": ["extracted_images/page_4_img_1.png","extracted_images/page_4_img_2.png"],
        "Verifying an order": ["extracted_images/page_4_img_3.png","extracted_images/page_4_img_4.png"]

    }
    return keyword_image_map

# Function to extract text and create embeddings from the PDF
def extract_text_and_create_embeddings(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./new_chroma_db"
    )
    vectordb.persist()
    
    return vectordb

vectordb = extract_text_and_create_embeddings(pdf_path)

# Function to call the LLM model via Bedrock
def invoke_llama_model(prompt_text):
    model_id = "meta.llama3-70b-instruct-v1:0"
    payload = {
        "prompt": prompt_text,
        "max_gen_len": 2000,
    }

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType='application/json',
        body=json.dumps(payload)
    )

    result = json.loads(response['body'].read().decode('utf-8'))
    return result['generation']

# Function to find the best matching keyword
def find_best_matching_keyword(user_query, keyword_image_map, threshold=0.6):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    user_query_embedding = model.encode([user_query])
    keyword_embeddings = model.encode(list(keyword_image_map.keys()))
    similarities = cosine_similarity(user_query_embedding, keyword_embeddings)[0]

    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[best_match_index]

    if best_match_similarity >= threshold:
        best_keyword = list(keyword_image_map.keys())[best_match_index]
        return best_keyword
    else:
        return None

# Function to display images in Streamlit
def display_images_in_streamlit(image_paths):
    for image_path in image_paths:
        if os.path.exists(image_path):
            st.image(image_path, caption=f"{image_path}", width=200, use_column_width=True)
        else:
            st.error(f"Image not found: {image_path}")

# Function to process the chatbot query
def chatbot(query, vectordb, keyword_image_map):
    if not query.strip():
        st.write("Please ask a valid question.")
        return

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    retrieved_docs = vectordb.similarity_search(query, k=4)
    relevant_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    best_keyword = find_best_matching_keyword(query, keyword_image_map)
    relevant_images = keyword_image_map.get(best_keyword, []) if best_keyword else []

    template = f"""
    You are a friendly, kind, and patient assistant for helpdesk. Act like a chatbot. Your task is to provide step-by-step instructions for user queries about using POET, based on the following content. Do not add any information that is not present in the given content. If the query does not match the context request the user to stay in context. Format your response as follows:
    1. Start the response by "Surely I can help you. Here are the steps:". 
    2. Then, ALWAYS provide a bullet point list of all the steps.
    3. Number each step and provide clear, concise instructions.
    4. If there are sub-steps, use indented bullet points.
    5. Use exactly the same wording and formatting as in the original instructions.
    6. DO NOT Skip any original instructions.

    Keep your answers to the point and don't include text from unrelated parts of the document
    Do not include phrases like "based on the context given" or "based on this line." or any explanation
    <>

    {relevant_content}

    {query}
    Possible Answer :
    """
    
    prompt_template = PromptTemplate(template=template, input_variables=["relevant_content", "query"])
    prompt = prompt_template.format(relevant_content=relevant_content, query=query)

    response = invoke_llama_model(prompt)
    return response, relevant_images

def greetings_generator(prompt):
    yield GREETINGS

if __name__ == '__main__':
    st.image("fedway-logo.png", use_column_width=False, width=300)
    st.title("Fedway Services - Helpdesk POC")
    # pdf_path = "POET_Everyday_Instructions_2page.pdf"
    keyword_image_map = create_keyword_mapping()
    # vectordb = extract_text_and_create_embeddings(pdf_path)
    st.markdown("<p style='font-size:14px; font-weight:bold;'>Hi! I am a chatbot to help you with POET Instructions.</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []

    # for i, message in enumerate(st.session_state.messages):
    #     with st.chat_message(message["role"]):
    #         if message["content"].startswith("Image reference: "):
    #             image_index = int(message["content"].split()[-1])  
    #             st.image(st.session_state.images[image_index], caption=f"Image")
    #         else:
    #             st.markdown(message["content"])

        # for i, message in enumerate(st.session_state.messages):
        #         with st.chat_message(message["role"]):
        #     st.markdown(message["content"])
        
        #     if message["content"].startswith("Image reference: "):
        #         image_index = int(message["content"].split()[-1])  
        #         st.image(st.session_state.images[image_index], caption=f"Image {image_index + 1}")

    if prompt := st.chat_input("What would you like to ask?"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response, image_paths = chatbot(prompt, vectordb, keyword_image_map)
            response = response.replace('\n', '<br>')
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response, unsafe_allow_html=True)

            for image_path in image_paths:
                if os.path.exists(image_path):
                    st.session_state.images.append(image_path)
                    image_index = len(st.session_state.images) - 1  
                    st.session_state.messages.append({"role": "assistant", "content": f"Image reference: {image_index}"})
                    st.image(image_path, caption=f"Image", width=200)
