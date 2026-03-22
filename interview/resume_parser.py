import pdfplumber
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -------- Extract text from resume -------- #

def extract_resume_text(file_path):

    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()

            if content:
                text += content + "\n"

    return text


# -------- Extract skills using Groq -------- #

def extract_skills(text):

    prompt = f"""
Extract only technical skills from the following resume.

Rules:
- Return only skills
- Comma separated
- Maximum 10 skills

Resume:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        skills = response.choices[0].message.content

    except Exception as e:

        print("Groq API error:", e)

        # fallback skills if API fails
        skills = "Python, SQL, Data Structures, Machine Learning"

    # convert to list
    skill_list = [s.strip().lower() for s in skills.split(",")]

    return skill_list
