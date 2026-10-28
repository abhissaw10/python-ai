from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-6-luna",
    instructions="You are a witty travel guide.",
    input="Suggest one thing to do in Bangalore.",
)
print(response.output_text)
