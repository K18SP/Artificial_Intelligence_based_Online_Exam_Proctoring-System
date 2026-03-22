from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_first_question(resume_text):

    prompt = f"""
    You are an AI interviewer.

    Resume:
    {resume_text}

    Ask the first technical interview question based on the resume skills.
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )

    return response.choices[0].message.content


def generate_followup_question(answer):

    prompt = f"""
    Candidate answered this in interview:

    {answer}

    Ask a deeper follow-up technical question.
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content