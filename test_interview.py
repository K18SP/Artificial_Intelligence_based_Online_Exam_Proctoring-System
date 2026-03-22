import time
import re
from interview.interview_manager import InterviewManager

manager = InterviewManager("RESUME_ONE.pdf")

question = manager.start_interview()

print("\nAI Interviewer:")
print(question)

# Interview settings
MAX_DURATION = 600      # 10 minutes
MAX_QUESTIONS = 8
MIN_QUESTIONS = 3
CONFIDENCE_TARGET = 70  # percent

start_time = time.time()

scores = []
question_count = 0


def extract_score(evaluation_text):
    """
    Extract score from AI evaluation text.
    Example: 'Score: 7/10'
    """
    match = re.search(r"(\d+)/10", evaluation_text)
    if match:
        return int(match.group(1))
    return 0


while True:

    if time.time() - start_time > MAX_DURATION:
        print("\nInterview ended: Time limit reached.")
        break

    answer = input("\nYour Answer: ")

    # Determine difficulty
    # Adaptive difficulty based on last answer
    if scores:
        last_score = scores[-1]
    else:
        last_score = 5

    if last_score <= 3:
        difficulty = "beginner"

    elif last_score <= 6:
        difficulty = "intermediate"

    else:
        difficulty = "advanced"

    evaluation, next_q = manager.next_step(answer, difficulty)

    print("\nEvaluation:")
    print(evaluation)

    score = extract_score(evaluation)
    scores.append(score)

    question_count += 1

    avg_score = sum(scores) / len(scores)
    confidence = avg_score * 10

    print(f"\nCurrent Confidence Score: {confidence:.2f}%")
    print("Current Difficulty Level:", difficulty)

    if confidence >= CONFIDENCE_TARGET and question_count >= MIN_QUESTIONS:
        print("\nInterview ended: Candidate reached confidence threshold.")
        break

    if question_count >= MAX_QUESTIONS:
        print("\nInterview ended: Maximum questions reached.")
        break

    print("\nNext Question:")
    print(next_q)


print("\nFinal Results")
print("----------------------")

print("Questions Answered:", question_count)
print("Average Score:", round(avg_score, 2))
print("Confidence Score:", round(confidence, 2), "%")

if confidence >= CONFIDENCE_TARGET:
    print("Result: Recommended")
else:
    print("Result: Needs Improvement")