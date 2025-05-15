from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from note_engine import note_engine
#from pdf import get_index, get_pdf_index_from_dict, get_pdf_index_from_set, get_pdf_index_from_list, get_pdf_index_from_string, get_pdf_index_from_tuple, get_pdf_index_from_frozenset, get_pdf_index_from_memoryview, get_pdf_index_from_bytearray, get_pdf_index_from_array, get_pdf_index_from_dataframe, get_pdf_index_from_series
from pdf import canada_engine
from prompt import new_prompt, coffee_context, instruction_str, context
from llama_index.core.tools import QueryEngineTool, ToolMetadata
#from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent


load_dotenv()

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df,
    VERBOSE=True,
    instruction_str=instruction_str,
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this tool can answer questions about world population statistics , demographics and details about a country.",
        ),
    ),
    QueryEngineTool(
        query_engine = canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this tool can answer questions about Canada, including its history, geography, and culture.",
        ),
    ),  
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="coffee_data",
            description=(
                "this tool can answer questions about coffee shops near National Gallery of Canada.",
                "The data includes shop name, address, price level (1-3), rating (1-5), and distance from the gallery."
            )
        ),
    ),      
]

#print(population_df.head())
llm = OpenAI(model="gpt-4o-mini-2024-07-18")#resp = llm.complete("Write a short, but joyous, ode to LlamaIndex")
#print(resp)

agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, context=context + "\n" + coffee_context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)