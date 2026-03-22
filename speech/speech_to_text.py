import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone()

# Calibrate once at startup
with mic as source:
    print("Calibrating microphone...")
    recognizer.adjust_for_ambient_noise(source, duration=2)

# ── Tuned parameters ─────────────────────────────────────────────────────────
#
# energy_threshold  : 300 (was 200) — slightly higher to ignore background hum
#                     without missing quiet voices.
#
# pause_threshold   : 2.0s (was 1.2s) — how long silence must last mid-speech
#                     before the recogniser decides the phrase is complete.
#                     1.2s is too short; natural speakers pause 1-2s between
#                     sentences. 2.0s gives room for thought without feeling slow.
#
# dynamic_energy_threshold : False — keeps energy_threshold stable so a
#                     loud environment doesn't suddenly mute the mic.
#
# phrase_time_limit : 25s per chunk (was 10s) — each call to listen() can
#                     capture up to 25 seconds of continuous speech.
#                     collect_full_answer() in main.py stitches chunks, so
#                     answers of any length are supported.
#
# timeout           : 8s (was 7s) — how long to wait for speech to BEGIN.

recognizer.energy_threshold         = 300
recognizer.pause_threshold          = 2.0
recognizer.dynamic_energy_threshold = False

LISTEN_TIMEOUT        = 8     # seconds to wait for speech to start
PHRASE_TIME_LIMIT     = 25    # seconds max per single chunk


def listen() -> str:
    """
    Capture one phrase (up to PHRASE_TIME_LIMIT seconds) and return the
    recognised text in lowercase.  Returns "" on silence, unintelligible
    audio, or API error.

    For long answers, call this repeatedly and stitch the results —
    see collect_full_answer() in main.py.
    """
    with mic as source:
        print("Listening... Speak now")
        try:
            audio = recognizer.listen(
                source,
                timeout          = LISTEN_TIMEOUT,
                phrase_time_limit= PHRASE_TIME_LIMIT,
            )
            text = recognizer.recognize_google(audio)
            print("Candidate said:", text)
            return text.lower()

        except sr.WaitTimeoutError:
            print("No speech detected")
            return ""

        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""

        except sr.RequestError as e:
            print(f"Speech API unavailable: {e}")
            return ""

        except Exception as e:
            print(f"listen() unexpected error: {e}")
            return ""