from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandanQueryEngine

from llama_index.query_engine import RetrieverQueryEngine


load_dotenv()

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandanQueryEngine(
    df=population_df,
    VERBOSE=True,
)

print(population_df.head())