import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

DOCGPT_OPENAI_API_KEY = os.getenv('DOCGPT_OPENAI_API_KEY')


def callopenai(userprompt, systemprompt):

    client = OpenAI(api_key=DOCGPT_OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.6,
        messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": userprompt},
        ]
    )
    return response.choices[0].message.content
