from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# To Access LLM API's of HuggingFace
llm = HuggingFaceEndpoint(
    # repo_id= "google/gemma-2-2b-it",
    # repo_id= "google/gemma-4-31B-it",
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.5
    )

model = ChatHuggingFace(llm=llm)

# Runnable Sequence
######################################################################
# class JokeOutput(BaseModel):
#     text: str = Field(..., description="The joke text")

# parser_1 = PydanticOutputParser(pydantic_object=JokeOutput)

# prompt_1 = PromptTemplate(template="Write a one liner joke on {topic}.\n{instructions}",
#                           input_variables=["topic"],
#                           partial_variables={"instructions": parser_1.get_format_instructions()})

# prompt_2 = PromptTemplate(template="Explain this joke {text}",
#                             input_variables=['text'])

# parser_2 = StrOutputParser()

# chain = RunnableSequence(prompt_1, model, parser_1, prompt_2, model, parser_2)

# response = chain.invoke({"topic": "Madhya Pradesh"})
# print(response)

# Runnable Parallel
######################################################################
# prompt1 = PromptTemplate(template="Write Twitter post for topic {topic}",
#                           input_variables=['topic'])

# prompt2 = PromptTemplate(template="Write LinkedIn post for topic {topic}",
#                           input_variables=['topic'])

# parser = StrOutputParser()

# chain = RunnableParallel(
#     {
#     "twitter": RunnableSequence(prompt1, model, parser),
#     "linkedin": RunnableSequence(prompt2, model, parser)
#     }
# )

# response = chain.invoke({"topic": "Artificial Intelligence"})
# print(response)

# Runnable Passthrough
######################################################################
# class JokeOutput(BaseModel):
#     text: str = Field(..., description="The joke text")

# parser_1 = PydanticOutputParser(pydantic_object=JokeOutput)

# prompt_1 = PromptTemplate(template="Write a one liner joke on {topic}.\n{instructions}",
#                           input_variables=["topic"],
#                           partial_variables={"instructions": parser_1.get_format_instructions()})

# prompt_2 = PromptTemplate(template="Explain this joke {text}",
#                             input_variables=['text'])

# parser_2 = StrOutputParser()

# joke_gen = RunnableSequence(prompt_1, model, parser_1)

# parallel_chain = RunnableParallel(
#     {
#         'joke': RunnablePassthrough(),
#         'explanation': RunnableSequence(prompt_2, model, parser_2)
#     }
# )

# chain = RunnableSequence(joke_gen, parallel_chain)

# response = chain.invoke({"topic": "Madhya Pradesh"})
# print(response)

# Runnable Lambda
######################################################################
# prompt1 = PromptTemplate(template="Write a small description for topic {topic}",
#                           input_variables=['topic'])

# parser = StrOutputParser()

# def word_count(text: str) -> int:
#     return len(text.split())

# topic_chain = RunnableSequence(prompt1, model, parser)

# parallel_chain = RunnableParallel(
#     {
#         'description': RunnablePassthrough(),
#         'word_count': RunnableLambda(word_count)
#     }
# )

# final_chain = RunnableSequence(topic_chain, parallel_chain)

# response = final_chain.invoke({"topic": "Artificial Intelligence"})

# print(response)

# Runnable Branch
#######################################################################
prompt1 = PromptTemplate(template="Write a report on topic {topic}",
                          input_variables=['topic'])

parser = StrOutputParser()

prompt2 = PromptTemplate(template="Summarise the following text\n {text}",
                          input_variables=['text'])

def word_count(text):
    return len(text.split())

report_chain = RunnableSequence(prompt1, model, parser)

conditional_chain = RunnableBranch(
    (lambda x: word_count(x)>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
    )

final_chain = RunnableSequence(report_chain, conditional_chain)

response = final_chain.invoke({"topic": "Artificial Intelligence"})
print(response)