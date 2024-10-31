from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain import LLMChain

# Step 1: Load a Small Dataset Using Streaming
dataset = load_dataset("trivia_qa", "rc", split="train[:100]", streaming=True)

# Step 2: Initialize Smaller Model Components
# Here, we use distilbert-base-uncased-distilled-squad, which is much smaller than RAG.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Set up the pipeline for question-answering
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Wrap model in LangChain-compatible pipeline
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Step 3: Define LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="Question: {query}\nContext: {context}\nAnswer:"
)
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# Step 4: Generate Response with Custom Function
def generate_response(query):
    # Streaming through a small subset of the dataset to find context for the query
    for i, entry in enumerate(dataset):
        # Using only the first context for simplicity
        context = entry["question"]
        answer = llm_chain.run(query=query, context=context)
        print("Context:", context)
        print("Answer:", answer)
        break  # Only using the first context to keep it short

# Test the pipeline
if __name__ == "__main__":
    query = "Explain retrieval-augmented generation."
    print("Response:", generate_response(query))

