import random
from interview.resume_parser import extract_resume_text, extract_skills
from interview.question_generator import generate_first_question
from interview.answer_analyzer import generate_followup_question
from interview.evaluator import evaluate_answer


class InterviewManager:

    def __init__(self, resume_path):

        resume_text = extract_resume_text(resume_path)

        self.skills = extract_skills(resume_text)

        self.used_skills = []

        self.current_skill = None
        self.current_question = None


    def select_new_skill(self):

        remaining = list(set(self.skills) - set(self.used_skills))

        if not remaining:
            self.used_skills = []
            remaining = self.skills

        skill = random.choice(remaining)

        self.used_skills.append(skill)

        return skill


    def start_interview(self):

        self.current_skill = self.select_new_skill()

        question = generate_first_question([self.current_skill])

        self.current_question = question

        return question


    def next_step(self, answer, difficulty):

        evaluation = evaluate_answer(self.current_question, answer)

        # extract score
        score = 5
        if "Score:" in evaluation:
            try:
                score = int(evaluation.split("Score:")[1].split("/")[0])
            except:
                score = 5

        # Topic logic
        if score >= 7:
            # continue same topic
            skill = self.current_skill
        else:
            # change skill
            skill = self.select_new_skill()
            self.current_skill = skill

        next_q = generate_followup_question(skill + " " + answer, difficulty)

        self.current_question = next_q

        return evaluation, next_q