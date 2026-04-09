from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# To Access LLM API's of HuggingFace
llm = HuggingFaceEndpoint(
    # repo_id= "google/gemma-2-2b-it",
    repo_id= "google/gemma-4-31B-it",
    task="text-generation",
    temperature=0.2
    )

model = ChatHuggingFace(llm=llm)

class TigerReserves(BaseModel):
    reserves: List[str] = Field(description="Name of Tiger Reserves in the state")

class TigerReserveInfo(BaseModel):
    name: str = Field(description="Name of the tiger reserve")
    established_year: str = Field(description="Year when the tiger reserve was established")

class TigerReserveInfoList(BaseModel):
    reserves_info: List[TigerReserveInfo] = Field(description="List of tiger reserves with their established year")

parser_1 = PydanticOutputParser(pydantic_object=TigerReserves)

prompt_1 = PromptTemplate(template="How many Tiger reserves are there in {state}?\n{instructions}",
                          input_variables=["state"],
                          partial_variables={"instructions": parser_1.get_format_instructions()})

parser_2 = PydanticOutputParser(pydantic_object=TigerReserveInfoList)

prompt_2 = PromptTemplate(template="When were these tiger reserves established? {reserves}?\n{instructions}",
                            input_variables=["reserves"],
                            partial_variables={"instructions": parser_2.get_format_instructions()})  

chain = prompt_1 | model | parser_1 | prompt_2 | model | parser_2

response = chain.invoke({"state": "Madhya Pradesh"})
print(response)