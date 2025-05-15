from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandanQueryEngine
from llama_index.query_engine import RetrieverQueryEngine
from note_engine import note_engine
from prompts import new_prompt, coffee_context, instruction_str, context
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.gemini import Gemini
from llama_index.core.agent import ReActAgent


load_dotenv()

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandanQueryEngine(
    df=population_df,
    VERBOSE=True,
    instruction_str=instruction_str,
)
population_query_engine = update_promps({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this tool can answer questions about world population statistics , demographics and details about a country.",
        ),
    )
]

print(population_df.head())

llm = Gemini(model="models/gemini-2.5-flash-preview-04-17")
#resp = llm.complete("Write a short, but joyous, ode to LlamaIndex")
#print(resp)

agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, context=context + "\n" + coffee_context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)