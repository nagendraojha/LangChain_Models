from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-3.5-turbo")
result=model.invoke("What is the capital of France?")
#print(result) # it will give answer in kwards arguments
#print(result['content']) # it will give answer in string
print(result.content) # it will give answer in string