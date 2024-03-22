from openai import OpenAI

client = OpenAI(api_key="sk")

response = client.images.generate(
  model="dall-e-3",
  prompt="the original bert the muppet from sesame street, but with a slight reference to healthcare and switzerland",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)