from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluate_answer(question, answer):

    prompt = f"""
    You are an AI technical interviewer.

    Question:
    {question}

    Candidate Answer:
    {answer}

    Evaluate the answer and provide:

    1. Score out of 10
    2. Short feedback
    3. Skill level (Beginner / Intermediate / Advanced)

    Format your response like:

    Score: X/10
    Feedback: ...
    Level: ...
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content