from openai import OpenAI
client = OpenAI()

# response = client.images.generate(
#     prompt="A cute baby sea otter",
#     n=2,
#     size="256x256"
# )
#
# print(response.data[0].url)

completion = client.chat.completions.create(
   model="gpt-3.5-turbo",
   messages=[
       {"role": "system", "content": "You are a helpful assistant."},
       {
           "role": "user",
           "content": "Write a haiku about recursion in programming."
       }
   ]
)

print(completion.choices[0].message)