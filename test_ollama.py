from ollama import chat
from ollama import ChatResponse

test_model = "deepseek-r1:8b"  # deepseek-r1:8b, deepseek-r1:1.5b, llama3.2

response: ChatResponse = chat(model=test_model, messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(f"MODEL: {test_model}")
print(response['message']['content'])
# or access fields directly from the response object
print(f":::::::::::::::::::::::::::::::::::::::::::\n{response.message.content}")
