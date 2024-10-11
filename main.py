import os
import json
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
from pandasai.llm.azure_openai import AzureOpenAI
from pandasai.prompts import AbstractPrompt
from pandasai import SmartDataframe
from pandasai.responses.streamlit_response import StreamlitResponse

# Load environment variables from .env file (Optional)
load_dotenv()

def main():
    
    # Set the title and subtitle of the app
    st.title('Chat With API Data')
    #st.subheader('Input api URL and create embedding, ask questions, and receive answers directly from the api response.')
    url = 'https://chebz162229094.sl2469408.sl.dst.ibm.com:444/en/shibumiaceapi/shibumiace'       
    #prompt='what is geograpy for solutiontitle SAP Basis Monitoring Self Heal Automation'
    prompt = st.text_input("Ask a question:") 

    # creating directory for embedding
    pklPath = os.getcwd()+"/embeddings"
    if not os.path.exists(pklPath):
        os.makedirs(pklPath)
    if os.path.isfile(f"{pklPath}/PKLData.pkl"):
        if st.button("Submit Query", type="primary"): 
            def get_agent_response(uploaded_file_content, query):
                class MyCustomPrompt(AbstractPrompt):
                    template = """Imagine you're a analyst and your task is to analyze the data. 
                    Filter the data based on certain criteria, analyze the data to identify trends or patterns, 
                    and present the extracted information in a clear and organized manner.
                    Make sure to verify the accuracy of the extracted data and provide insights or recommendations based on your analysis
                                You have been provided with a dataframe:{dataframes}                        
                            return string with:
                                    - type (possible value: "text")
                                    - value (string)
                                    {{ "type": "string", "value": "This is text" }}
                                    or
                                    {{ "type": "dataframe", "value": pd.DataFrame({{...}}) }}
                                
                                        ``python
                                def analyze_data(dfs: list[pd.DataFrame]) -> dict:
                                # Code goes here (do not add comments)

                                # Declare a result variable and put insights in it
                                result = analyze_data(dfs)
                                ``
                                Using the provided dataframes (`dfs`), update the Python code based on the user's query:
                                {conversation}
                                # Updated code:
                                # """

                AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
                AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
                AZURE_OPENAI_API_BASE = os.getenv('AZURE_OPENAI_API_BASE')
                AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

                llm = AzureOpenAI(api_token=AZURE_OPENAI_API_KEY,
                          api_version=AZURE_OPENAI_API_VERSION,
                          api_base=AZURE_OPENAI_API_BASE,
                          deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                          )
                # st.write(llm)
                # dfnew = pandas.DataFrame(uploaded_file_content)
                df = SmartDataframe(uploaded_file_content, config={"llm": llm, "enable_cache": False, "verbose": True, "max_retries": 3,
                                                                "is_chat_model": True,  "response_parser": StreamlitResponse,
                                                                "custom_prompts": {"generate_python_code": MyCustomPrompt()}})

                response = df.chat(query)
                return response
            if prompt.strip():
                df=pd.read_pickle(f"{pklPath}/PKLData.pkl")                
                result_text= get_agent_response(df, prompt)
                if type(result_text) is list:
                        st.write("")
                        for i in range(len(result_text)):
                            for key in result_text[i]:
                                if result_text[i] == "value":
                                    st.write(result_text[i][key])
                else:
                        if (result_text != None):
                            st.write("")
                            st.write(result_text)   
            else:
                st.write("Please enter value in ask a question")
       
    else:                
        response = requests.get(url, verify=False)
        with st.spinner("Fetching API data..."):            
            if response.status_code == 200:
                data_json = response.json()
                data = json.dumps(data_json)                
                dirPath = os.getcwd()                
                # write json to a file
                with open(f"{dirPath}/Data.json", "w") as f:
                    f.write("".join(data))               
                
                # reading json and creating pkl file
                df = pd.read_json(f"{dirPath}/Data.json")
                df.to_pickle(f"{pklPath}/PKLData.pkl")                
                st.write("Fetched data successfully. Please ask Query")
            else:
                st.error('Failed to retrieve data')
        
      

if __name__ == '__main__':
    main()