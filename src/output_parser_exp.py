from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# To Access LLM API's of HuggingFace
llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.5
    )

model = ChatHuggingFace(llm=llm)
###############################################################
# template1 = PromptTemplate(template="Write a report on {topic}",
#                            input_variables=["topic"])

# template2 = PromptTemplate(template="Create 5 line summary of following report\n {report}",
#                            input_variables=["report"])

# parser = StrOutputParser()

# chain = template1 | model | parser | template2 | model | parser

# response = chain.invoke({"topic": "Quantum Computing"})
###############################################################
# parser = JsonOutputParser()

# template = PromptTemplate(template="Give me names of last 5 prime ministers of {country} with tenure.\n {instructions}",
#                           input_variables=["country"],
#                           partial_variables={"instructions": parser.get_format_instructions()})

# chain = template | model | parser

# response = chain.invoke({"country": "India"})

##############################################################
# schema = [
#     ResponseSchema(name='fact_1', description='fact1 about prime minister'),
#     ResponseSchema(name='fact_2', description='fact2 about prime minister'),
#     ResponseSchema(name='fact_3', description='fact3 about prime minister')
# ]

# parser = StructuredOutputParser.from_response_schemas(schema)

# template = PromptTemplate(template="Give me names of 3 facts about prime ministers of {country}\n {instructions}",
#                           input_variables=["country"],
#                           partial_variables={"instructions": parser.get_format_instructions()})

# chain = template | model | parser

# response = chain.invoke({"country": "India"}) 

################################################################

class PrimeMinister(BaseModel):
    Name: str = Field(description="Name of prime minister")
    Tenure: str = Field(description="Tenure of prime minister")
    Party_Name: str = Field(description="Party Name of prime minister")

class PrimeMinisters(BaseModel):
    primeministers : List[PrimeMinister] = Field(description="Detail of the Prime Minister")

parser = PydanticOutputParser(pydantic_object=PrimeMinisters)

template = PromptTemplate(template="Give me names of last 5 prime ministers of {country} with Tenure and Party he belongs to.\n {instructions}",
                          input_variables=["country"],
                          partial_variables={"instructions": parser.get_format_instructions()})

chain = template | model | parser

response = chain.invoke({"country": "India"}) 

print(response.model_dump())