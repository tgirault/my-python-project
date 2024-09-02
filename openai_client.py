from openai import OpenAI
client = OpenAI(
  organization='org-xxx',
  project='proj_xxx',
  api_key='sk-proj-xxx',
)
response = client.images.generate(
    model="dall-e-3",
    prompt="Could you generate a garden gnome like King Kong at the top of the Eiffel Tower?",
    n=1,
    size="1024x1792",
)


print(response.data[0].url)

completion = client.chat.completions.create(
   model="gpt-4o-mini",
   messages=[
       {"role": "system", "content": "You are a helpful assistant."},
       {
           "role": "user",
           "content": "Could you tell me why the sea is salted?",
       }
   ]
)

print(completion.choices[0].message)