from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
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

class TigerReserves(BaseModel):
    reserves: List[str] = Field(description="Name of Tiger Reserves in the state")

parser_1 = PydanticOutputParser(pydantic_object=TigerReserves)

prompt_1 = PromptTemplate(template="How many Tiger reserves are there in {state}?\n{instructions}",
                          input_variables=["state"],
                          partial_variables={"instructions": parser_1.get_format_instructions()})

prompt_2 = PromptTemplate(template="Get the Number of Tiger in tiger reserves {reserves}?",
                            input_variables=["reserves"])  

chain = prompt_1 | model | parser_1 | prompt_2 | model | StrOutputParser()

response = chain.invoke({"state": "Madhya Pradesh"})
print(response)