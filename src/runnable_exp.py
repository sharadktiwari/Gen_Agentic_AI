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

prompt_1 = PromptTemplate(template="Write a one liner joke on {topic}",
                          input_variables=["topic"])

parser = StrOutputParser()

# chain = prompt_1 | model | parser
chain = RunnableSequence(prompt_1, model, parser)

response = chain.invoke({"topic": "Madhya Pradesh"})
print(response)