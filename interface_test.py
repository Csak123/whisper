import requests
import json

url = "http://localhost:5000/transcribe"
data = {
    "audio_path": "D:\\Projects\\PycharmProjects\\Whisper-Finetune\\dataset\\test.wav"
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
