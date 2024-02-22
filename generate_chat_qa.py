import os

from question_generation.chat_QA_generator.gemini import GeminiChatQAGenerator
from datasets import Dataset
from dotenv import load_dotenv

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# generate chat-like questions and answers using the Gemini model
if __name__ == "__main__":
    dataset_name = 'felipeoes/br_federal_legislation'
    
    generator = GeminiChatQAGenerator(dataset_name)
    result = generator.run(sample=10) # sample = 10 for testing purposes
    
    # first 10 examples
    print(result)
    print(result[:10])
    
    # upload the result to the Hugging Face model hub
    result.push_to_hub('felipeoes/br_federal_legislation_qa', token=HUGGINGFACE_TOKEN)
     
     