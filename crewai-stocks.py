
#import das libs
import json
import os

from datetime import datetime
import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_cummunity.tools import DuckDuckGoSearchResults

from streamlit as st


# In[ ]:


def fetch_stock_price(ticket):
    stock = yf.dowload(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} form the last year about a specificcompany form Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)


# In[ ]:


#importando openia llm-gpt
os.environ['OPENIA_API_KEY'] = st.secrets['OPENIA_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[ ]:


# importando a ferramenta de pesquisa
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


# In[ ]:


stockPriceAnalyst = Agent(
    role="",
    gool="",
    backstory="",
    verbose=True,
    llm=llm,
    max_iter= 5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)


# In[ ]:


getStockPrice = Task(
    description="",
    expected_output="",
    agent= stockPriceAnalyst
)


# In[ ]:


newsAnalyst = Agent(
    role="",
    gool="",
    backstory="",
    verbose=True,
    llm=llm,
    max_iter= 10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False
)


# In[ ]:


get_news = Task(
    description="",
    expected_output="",
    agent= newsAnalyst
)


# In[ ]:


stockAnalystWrite = Agent(
    role="",
    gool="",
    backstory="",
    verbose=True,
    llm=llm,
    max_iter= 5,
    memory=True,
    allow_delegation=True
)


# In[ ]:


writeAnalyses = Task(
    description="",
    expected_output="",
    agent= stockAnalystWrite,
    context=[getStockPrice, get_news]
)


# In[ ]:


crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose=2,
    process= Process.hierachical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)


# In[ ]:

with st.sidebar:
    st.header("")
    with st.form(key='researc_form'):
        topic = st.text_input("")
        submit_button = st.form_submit_button(label = "")
if submit_button:
    if not topic:
        st.error("")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("")
        st.write(results['final_output'])

