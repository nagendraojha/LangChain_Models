from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/chat-bison-001")
response = llm.invoke(HumanMessage(content="Tell me a joke."))
print(response.content)
