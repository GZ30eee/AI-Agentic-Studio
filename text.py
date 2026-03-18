from google import genai  # pip install google-genai
client = genai.Client(api_key="AIzaSyC49K2raOxrPjfpURfVu_BFQ2dMmTHN_ek")  # Or use env var GEMINI_API_KEY
response = client.models.generate_content(model="gemini-2.5-flash", contents="Test")
print(response.text)
