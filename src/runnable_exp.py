from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# To Access LLM API's of HuggingFace
llm = HuggingFaceEndpoint(
    # repo_id= "google/gemma-2-2b-it",
    repo_id= "google/gemma-4-31B-it",
    task="text-generation",
    temperature=0.5
    )

model = ChatHuggingFace(llm=llm)

class JokeOutput(BaseModel):
    text: str = Field(..., description="The joke text")

parser_1 = PydanticOutputParser(pydantic_object=JokeOutput)

prompt_1 = PromptTemplate(template="Write a one liner joke on {topic}.\n{instructions}",
                          input_variables=["topic"],
                          partial_variables={"instructions": parser_1.get_format_instructions()})

prompt_2 = PromptTemplate(template="Explain this joke {text}",
                            input_variables=['text'])

parser_2 = StrOutputParser()

# chain = prompt_1 | model | parser
chain = RunnableSequence(prompt_1, model, parser_1, prompt_2, model, parser_2)

response = chain.invoke({"topic": "Madhya Pradesh"})
print(response)
