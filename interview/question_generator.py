from groq import Groq
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_first_question(skills):

    skill_text = ", ".join(skills)

    prompt = f"""
You are an AI interviewer.

Ask ONE technical interview question about this skill:

{skill_text}

Rules:
- maximum 20 words
- one question only
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content
# Create Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_first_question(resume_text):

    prompt = f"""
    You are an AI interviewer.

    Based on the resume below, ask ONLY ONE short technical interview question.

    Rules:
    - Ask only ONE question
    - Maximum 20 words
    - Do NOT include explanation

    Resume:
    {resume_text}
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content