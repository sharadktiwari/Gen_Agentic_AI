from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# To Access LLM API's of HuggingFace
llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.5
    )

model = ChatHuggingFace(llm=llm)

###################################################################################################
# Simple Chain
# prompt_1 = PromptTemplate(template="How many Tiger reserves are there in {state}?",
#                           input_variables=["state"])

# prompt_2 = PromptTemplate(template="Get the Number of Tiger in tiger reserves {reserves}?",
#                             input_variables=["reserves"])  

# parser = StrOutputParser()

# chain = prompt_1 | model | parser | prompt_2 | model | parser

# response = chain.invoke({"state": "Madhya Pradesh"})
# print(response)


###################################################################################################
# Parallel Chain
# prompt_1 = PromptTemplate(template="Generate short and simple notes from following text\n {text}",
#                           input_variables=["text"])

# prompt_2 = PromptTemplate(template="Generate 5 short question answers from the following text\n {text}",
#                             input_variables=["text"])

# prompt_3 = PromptTemplate(template="Merge the provided notes and question answers into a single document.\n Notes: {notes}\n Question Answers: {qna}",
#                           input_variables=["notes", "qna"])

# parser = StrOutputParser()

# paraller_chain = RunnableParallel({
#     "notes": prompt_1 | model | parser,
#     "qna": prompt_2 | model | parser
#     })

# merge_chain = prompt_3 | model | parser

# chain = paraller_chain | merge_chain

# response = chain.invoke({"text": "Tiger reserves are protected areas in India established to conserve the Bengal tiger and its habitat. These reserves are part of Project Tiger, launched in 1973 by the Government of India to address the declining population of tigers in the country. Tiger reserves are managed by the National Tiger Conservation Authority (NTCA) and are spread across various states in India. They provide a safe haven for tigers and other wildlife, promoting biodiversity conservation and eco-tourism. Some well-known tiger reserves in India include Jim Corbett National Park, Ranthambore National Park, Bandhavgarh National Park, and Kanha National Park."})

# print(response)
###################################################################################################
# Conditional Chain
parser = StrOutputParser()

class SentimentOutput(BaseModel):
    sentiment: Literal['pos', 'neg'] = Field(description="Give the sentiment of the feedback")

parser_2 = PydanticOutputParser(pydantic_object=SentimentOutput)

prompt_1 = PromptTemplate(template="Classify the sentiment of the following feedback text into 'pos' or 'neg'.\n Feedback: {feedback} \n {instructions}",
                          input_variables=["feedback"],
                          partial_variables={"instructions": parser_2.get_format_instructions()})

prompt_2 = PromptTemplate(template="Generate an appropriate response for this positive feedback\n feedback: {feedback}",
                            input_variables=["feedback"])

prompt_3 = PromptTemplate(template="Generate an appropriate response for this negative feedback\n feedback: {feedback}",
                            input_variables=["feedback"])

sentiment_chain = prompt_1 | model | parser_2

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'pos', prompt_2 | model | parser),
    (lambda x: x['sentiment'].sentiment == 'neg', prompt_3 | model | parser),
    RunnableLambda(lambda x: "No valid sentiment found")
    )

chain = {
    "feedback": RunnablePassthrough(),
    "sentiment": sentiment_chain
    } | branch_chain

response = sentiment_chain.invoke({"feedback": "This is a bad product."})
print(response)
#######################################################################################################################
# chain.get_graph().print_ascii()