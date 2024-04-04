import requests
import json

OPENAI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
OPENAI_API_KEY = 'API_KEY_DO_NOT_MODIFY'

def fetch_chat_completion(messages):
  headers = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'Cintent-Type': 'application/json' 
  }
  data = {
    'model':'gpt-3.5-turbo',
    'messages': messages,
    'max_tokens':150,
    'temperature':0.1
  }
  response = requests.post(OPENAI_API_ENDPOINT, headers=headers, json=data)
  if response.status_code == 200:
    return response.json()['choices'][0]['message']['content']
  else:
    raise Exception("Error fetching completion: " + response.text)

def main():
  messages = [{'role': 'user', 'content': 'I plan to visit Paris. Can you recommend a top sightseeing spot?'}]
  recommendations_requests = [
    'Great! Can you recommend another sightseeing spot?',
    'Awesome! Any more recommendations?',
  ]
  try:
    for request in recommendations_requests:
      assistant_response= fetch_chat_completion(messages)
      messages.append({'role':'assistant','content':assistant_response.strip()})
      messages.append({'role':'user','content':request})
  except Exception as e:
    print(str(e))

  print(json.dumps(messages, indent=4))

if __name__ == '__main__':
  main()
