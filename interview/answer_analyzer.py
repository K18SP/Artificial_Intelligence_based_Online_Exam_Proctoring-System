from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_followup_question(context, difficulty):

    prompt = f"""
You are an AI interviewer.

Context: {context}

Ask ONE {difficulty} level interview question.

Maximum 20 words.
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content