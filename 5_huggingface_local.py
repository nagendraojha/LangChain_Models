from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch

load_dotenv()

# Load model and tokenizer locally
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create pipeline with proper generation settings
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

# Create LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Generate response with proper prompt formatting
prompt = """<|system|>
You are a helpful AI assistant.</s>
<|user|>
What is the capital of India?</s>
<|assistant|>
"""
result = llm(prompt, max_new_tokens=100)

print("Generated response:")
print(result)