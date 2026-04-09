from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.load import loads

load_dotenv()

# To Access LLM API's of HuggingFace
llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    repo_id= "google/gemma-4-31B-it",
    # repo_id= "google/gemma-2-2b-it",
    task="text-generation")

# To Access Local LLM
# google/gemma-3-270m-it
# TinyLlama/TinyLlama-1.1B-Chat-v1.0

# llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#                                         task="text-generation",
#                                         # pipeline_kwargs=dict(
#                                         model_kwargs=dict(
#                                             temperature=0.5,
#                                             # max_new_tokens=512,
#                                             device_map="cpu"
#                                         ))

model = ChatHuggingFace(llm=llm)

with open('template.json', 'r') as f:
    template = loads(f.read())

# prompt = template.invoke({"paper_input": "Attention is all you need", "style_input": "Simple", "length_input": "Short"})

# response = model.invoke(prompt)

chain = template | model

response = chain.invoke({"paper_input": "Attention is all you need"})#, "style_input": "Simple", "length_input": "Short"})

print(response)