import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun

# 1. Load Model & Adapter
base_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# Replace with your actual repo ID after you push your fine-tuned model
adapter_id = "Prathamesh25/smollm2-aptitude-agent" 

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Load your fine-tuned weights onto the base model
model = PeftModel.from_pretrained(base_model, adapter_id)
model = model.merge_and_unload() # Optional: merges for slightly faster inference

# 2. Setup LangChain
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)
tools = [DuckDuckGoSearchRun()]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)

# 3. Gradio Logic
def chat_with_agent(message, history):
    # The history param is provided by Gradio but simplified here for the agent
    response = agent.run(message)
    return response

demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="SmolLM2 Aptitude Agent",
    description="Ask me technical aptitude questions or general queries!",
    examples=["What is the output of `int x=5; printf('%d', x++);`?", "Search for latest Python 3.12 features"]
)

if __name__ == "__main__":
    demo.launch()
