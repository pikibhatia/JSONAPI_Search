import os
import bs4
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma.vectorstores import Chroma

# Load environment variables from .env file (Optional)
load_dotenv()

# Define the API endpoint

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def main():
    # Set the title and subtitle of the app
    st.title('Chat With API Data')
    st.subheader('Input api URL, ask questions, and receive answers directly from the api response.')

    url = st.text_input("Insert The API URL")

    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):
        # Define the API endpoint
        #url = 'https://api.nationalize.io?name=nathaniel'

        # Make the API call
        response = requests.get(url)
        # Display the response in the app
        if response.status_code == 200:
            data = response.json()
            #st.write(data)
            ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
            DB_DIR: str = os.path.join(ABS_PATH, "db")

            # Load data from the specified URL
            loader = WebBaseLoader(url)
            data = loader.load()

            # Split the loaded data
            text_splitter = CharacterTextSplitter(separator='\n', 
                                    chunk_size=2000, 
                                    chunk_overlap=100,
                                    length_function=len)

            docs = text_splitter.split_documents(data)

            # Create OpenAI embeddings
            
            AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
            AZURE_OPENAI_DEPLOYMENT_NAME = "text-embedding-3-large"
            AZURE_MODEL_NAME = "text-embedding-3-large"
            AZURE_OPENAI_API_BASE = os.getenv('AZURE_OPENAI_API_BASE')
            AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

            openai_embeddings = AzureOpenAIEmbeddings(azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
                model=AZURE_MODEL_NAME,
                azure_endpoint=AZURE_OPENAI_API_BASE,
                openai_api_type="azure",
                api_version=AZURE_OPENAI_API_VERSION,
                openai_api_key=AZURE_OPENAI_API_KEY)

            # Create a Chroma vector database from the documents
            vectordb = Chroma.from_documents(documents=docs, 
                                    embedding=openai_embeddings,
                                    persist_directory=DB_DIR)

            # Create a retriever from the Chroma vector database
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            # Use a AzureChatOpenAI model
            llm = AzureChatOpenAI(api_key="69c2a3bc8924440a9886654cd40cce5a",
                        openai_api_version="2023-07-01-preview",
                        azure_endpoint="https://azureibmopenaigpt4.openai.azure.com/",
                        azure_deployment= "gpt35",
                        )

            # Create a RetrievalQA from the model and retriever
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # Run the prompt and get the response
            response = qa(prompt)
            st.write ("Output:")
            st.write(response["result"])
            
        else:
            st.error('Failed to retrieve data')
  
   

if __name__ == '__main__':
    main()