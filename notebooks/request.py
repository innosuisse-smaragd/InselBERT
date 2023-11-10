import requests

response = requests.post(
   "http://0.0.0.0:3000/extract_facts_and_anchors",
   headers={
      "accept": "text/plain",
      "Content-Type": "text/plain",
   },
   data="Kein Herdbefund festgestellt. Im linken Quadranten keine Verkalkungen."
)

print(response.text)