from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
summary = None

# Design and configuration of Streamlit
st.set_page_config(layout='wide')

st.title("Text Summarizer / Paraphraser")
st.divider()
col1, col2 = st.columns(2, gap='medium')
input_text  = col1.text_area("Enter the text", value=None, height=300)
col1.divider()
submit = col1.button("Summarize")
para = col1.button("Paraphrase")
col2.subheader("Output")
container = col2.container(border=True)
# ----------------------------------------------------
GOOGLE_API_KEY = "#YOUR API KEY"
llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY, temperature=0.6)

if submit:
    prompt_template = PromptTemplate(
        input_variables= ["text", "count"],
        template = "Summarize the given {text} strictly around {count} words"
    )

    chain = LLMChain(llm = llm, prompt=prompt_template)
    summary = chain.run({'text': input_text, "count": 50})
    #col2.text_area(label="Output", value = summary, height=450)
    
    container.write(summary)
elif para:

    prompt_template = PromptTemplate(
        input_variables= ["text"],
        template = "Paraphrase the given {text}"
    )

    chain = LLMChain(llm = llm, prompt=prompt_template)
    summary = chain.run(input_text)
    #col2.text_area(label="Output", value = summary, height=450)
    container.write(summary)
    




