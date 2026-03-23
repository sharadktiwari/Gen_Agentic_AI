from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(model_id="google/gemma-3-270m-it",
                                        task="text-generation",
                                        device_map = "cpu",
                                        pipeline_kwargs=dict(
                                        # model_kwargs=dict(
                                            temperature=0.5,
                                            max_new_tokens=50,
                                        ))

model = ChatHuggingFace(llm=llm)

# chat_template = ChatPromptTemplate([
#     ('system',"You are a helpful {domain} expert."),
#     ('user', "Explain about {topic}.")
# ])

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful support expert."),
    MessagesPlaceholder(variable_name="history"),
    ('user', "{query}")
])

chat_history = []

with open("chat_history", 'r') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'history': chat_history, 'query': "When I will get my refund?"})

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    print("AI: ", response.content)
    chat_history.append(AIMessage(content=response.content))

print(chat_history)