import os

from question_generation.chat_QA_generator.gemini import GeminiChatQAGenerator
from datasets import Dataset
from dotenv import load_dotenv

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# generate chat-like questions and answers using the Gemini model
if __name__ == "__main__":
    dataset_name = 'felipeoes/br_federal_legislation'
    
    generator = GeminiChatQAGenerator(dataset_name, verbose=True, num_api_keys=6) # there will be allocated 20 workers per API key
    result = generator.run(start=0)
    
    # first 3 examples
    print(result)
    print(result[:3])
    
    # upload the result to the Hugging Face model hub
    result.push_to_hub('felipeoes/br_federal_legislation_qa', token=HUGGINGFACE_TOKEN)