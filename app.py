import cv2
import time
import threading
import platform
from datetime import datetime
import dlib
import re

from proctoring.facial_detections import detectFace
from proctoring.blink_detection import isBlinking
from proctoring.eye_tracker import gazeDetection
from proctoring.mouth_tracking import mouthTrack
from proctoring.object_detection import detectObject
from proctoring.head_pose_estimation import head_pose_detection

from speech.speech_to_text import listen
from speech.text_to_speech import speak
from interview.interview_manager import InterviewManager
from cheating_detector import CheatingDetector


# ── Cross-platform beep ───────────────────────────────────────────────────────

def beep_signal():
    print("\n🎤 Speak now...")
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(2500, 400)


# ── Globals ───────────────────────────────────────────────────────────────────

data_record = []
look_away_start    = None
look_away_duration = 0.0
LOOK_AWAY_LIMIT    = 3
last_warning_time  = 0.0
face_history       = []

manager       = None
manager_ready = threading.Event()

listening_status   = "Idle"
current_question   = ""
current_score      = "-"         # FIX: shown as "-" until first answer scored
current_difficulty = "Beginner"
difficulty         = "beginner"

_last_objectName = []
_last_headPose   = "Center"

# Focus / clipboard (edge-triggered, protected by _focus_lock)
_window_focused = True
_tab_switched   = False
_clipboard_used = False
_prev_focused   = True
_focus_lock     = threading.Lock()

detector = CheatingDetector()

# Head-pose calibration status (set by head_pose_estimation module)
_pose_calibrated = False
_calib_frame_count = 0
CALIB_TOTAL = 30    # must match head_pose_estimation.CALIB_FRAMES


# ── Text helpers ──────────────────────────────────────────────────────────────

def sanitise_question(text: str) -> str:
    if not text:
        return text
    # Strip LLM meta-prefixes like "Here's a beginner-level question:?"
    text = re.sub(
        r"^(here'?s?\s+a\s+\w[\w\s\-]*?(question|interview question)\s*:?\s*\??)+",
        "", text, flags=re.IGNORECASE
    ).strip()
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def wrap_question(text: str, max_chars: int = 38) -> list:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line = (line + " " + word).strip()
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines[:4]


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera(index=0, retries=3):
    for attempt in range(retries):
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Camera opened on index {index} (attempt {attempt + 1})")
            return cam
        cam.release()
        time.sleep(1)
    for fb in range(1, 4):
        cam = cv2.VideoCapture(fb)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Camera opened on fallback index {fb}")
            return cam
        cam.release()
    raise RuntimeError("Could not open any camera.")


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid_dlib_rect(face) -> bool:
    try:
        return isinstance(face, dlib.rectangle) and face.width() > 0 and face.height() > 0
    except Exception:
        return False


def faceCount_detection(faceCount: int) -> str:
    global last_warning_time
    t = time.time()
    if faceCount > 1:
        if t - last_warning_time > 2:
            print("WARNING: Multiple faces detected")
            last_warning_time = t
        return "Multiple faces detected"
    if faceCount == 0:
        if t - last_warning_time > 2:
            print("WARNING: No face detected")
            last_warning_time = t
        return "No face detected"
    return "Face detecting properly"


# ── Focus / clipboard listeners ───────────────────────────────────────────────

def _start_focus_listener():
    global _window_focused, _tab_switched, _prev_focused
    try:
        import win32gui
        OUR_TITLE = "AI Interview System"
        while True:
            time.sleep(0.4)
            hwnd    = win32gui.GetForegroundWindow()
            focused = OUR_TITLE in win32gui.GetWindowText(hwnd)
            with _focus_lock:
                if _prev_focused and not focused:
                    _window_focused = False
                    _tab_switched   = True
                    detector.notify_tab_switch()
                    print("WARNING: Window lost focus")
                elif not _prev_focused and focused:
                    _window_focused = True
                _prev_focused = focused
    except ImportError:
        print("INFO: pywin32 not installed — tab-switch detection disabled. pip install pywin32")
    except Exception as e:
        print(f"WARNING: Focus listener: {e}")


def _start_clipboard_listener():
    global _clipboard_used
    try:
        import win32clipboard
        last_seq = 0
        while True:
            time.sleep(0.5)
            try:
                win32clipboard.OpenClipboard()
                seq = win32clipboard.GetClipboardSequenceNumber()
                win32clipboard.CloseClipboard()
                if seq != last_seq and last_seq != 0:
                    with _focus_lock:
                        _clipboard_used = True
                    detector.notify_clipboard()
                    print("WARNING: Clipboard activity detected")
                last_seq = seq
            except Exception:
                pass
    except ImportError:
        print("INFO: Clipboard monitoring requires pywin32.")
    except Exception as e:
        print(f"WARNING: Clipboard listener: {e}")


# ── Interview threads ─────────────────────────────────────────────────────────

def start_interview_async():
    global manager, current_question
    try:
        manager = InterviewManager("RESUME_ONE.pdf")
        question = manager.start_interview()
        current_question = sanitise_question(question) if question else "Tell me about yourself."
        print("\nAI Interviewer:", current_question)
        speak(current_question)
    except Exception as e:
        print(f"ERROR: InterviewManager failed: {e}")
        current_question = "Tell me about yourself."
        speak(current_question)
    finally:
        manager_ready.set()


def collect_full_answer(
    max_duration:     int = 120,  # hard ceiling in seconds
    max_empty_chunks: int = 3,    # consecutive empty chunks with NO prior speech → skip
    max_empty_after:  int = 2,    # consecutive empty chunks AFTER some speech → done
) -> str:
    """
    Stitches multiple listen() calls into one complete answer.

    How it ends:
      1. Candidate says "done" / "finish" / "next" / "skip"  → immediate stop
      2. max_empty_after consecutive empty chunks AFTER capturing some speech
         (each empty chunk = ~8s timeout, so 2×8 = ~16s of silence = done)
      3. max_empty_chunks consecutive empties with NO speech yet → skip question
      4. max_duration seconds total elapsed → force stop

    listen() is now tuned to:
      - phrase_time_limit = 25s  (captures long continuous sentences)
      - pause_threshold   = 2.0s (waits 2s of silence before cutting phrase)
      - timeout           = 8s   (waits up to 8s for speech to begin)

    So a candidate who pauses 1-2s mid-sentence is NOT cut off mid-thought.
    """
    global listening_status

    chunks:       list[str] = []
    empty_streak: int       = 0
    session_start = time.time()

    STOP_WORDS = {"done", "finish", "finished", "next", "next question", "skip"}

    while True:
        elapsed = time.time() - session_start
        if elapsed >= max_duration:
            print(f"Max duration ({max_duration}s) reached — finalising.")
            break

        try:
            chunk = listen()
        except Exception as e:
            print(f"WARNING: listen() exception: {e}")
            chunk = ""

        if chunk and chunk.strip():
            chunk        = chunk.strip()
            empty_streak = 0   # reset on any successful capture

            # Explicit stop word check
            if any(sw in chunk.lower() for sw in STOP_WORDS):
                print("Candidate signalled end of answer.")
                # Keep the chunk only if it has more than just the stop word
                non_stop = " ".join(
                    w for w in chunk.split()
                    if w.lower() not in STOP_WORDS
                )
                if non_stop:
                    chunks.append(non_stop)
                break

            chunks.append(chunk)
            combined   = " ".join(chunks)
            word_count = len(combined.split())
            print(f"  [{int(elapsed)}s] Chunk captured: '{chunk}'  ({word_count} words total)")
            listening_status = f"Listening ({word_count}w)"

        else:
            empty_streak += 1
            has_speech = len(chunks) > 0

            limit = max_empty_after if has_speech else max_empty_chunks
            print(f"  [{int(elapsed)}s] No speech — streak {empty_streak}/{limit}"
                  f"{'  (will finalise)' if has_speech else '  (will skip)'}")

            if empty_streak >= limit:
                if has_speech:
                    print("Silence detected — finalising answer.")
                else:
                    print("No speech at all — skipping question.")
                break

    return " ".join(chunks)


def interview_process():
    global difficulty, listening_status, current_question, current_score, current_difficulty

    print("Waiting for manager to be ready...")
    manager_ready.wait()
    print("Manager ready — starting interview loop.")

    while True:
        # Brief pause after question is spoken so TTS doesn't get picked up
        time.sleep(2)
        beep_signal()
        listening_status = "Listening"

        answer = collect_full_answer(
            max_duration     = 120,  # 2 min hard ceiling
            max_empty_chunks = 3,    # 3 × 8s = ~24s with no speech → skip
            max_empty_after  = 2,    # 2 × 8s = ~16s silence after speech → done
        )

        if not answer:
            print("Skipping — no response.")
            speak("I did not hear a response. Moving to the next question.")
            listening_status = "Idle"
            time.sleep(2)
            continue

        print(f"\nFull answer ({len(answer.split())} words): {answer}")
        listening_status = "Processing"

        if manager is None:
            listening_status = "Idle"
            continue

        try:
            evaluation, next_q = manager.next_step(answer, difficulty)
        except Exception as e:
            print(f"WARNING: next_step failed: {e}")
            speak("Issue evaluating your answer. Please try again.")
            listening_status = "Idle"
            continue

        print("\nEvaluation:", evaluation)

        try:
            parsed = int(evaluation.split("Score:")[1].split("/")[0].strip())
            current_score = parsed
        except (IndexError, ValueError):
            print("WARNING: Could not parse score — keeping previous value.")

        score_int = current_score if isinstance(current_score, int) else 5
        if   score_int <= 3: difficulty = "beginner"
        elif score_int <= 6: difficulty = "intermediate"
        else:                difficulty = "advanced"
        current_difficulty = difficulty

        if next_q and next_q.strip():
            current_question = sanitise_question(next_q)
            print("\nNext Question:", current_question)
            speak(current_question)

        listening_status = "Idle"
        time.sleep(3)


# ── Dashboard drawing ─────────────────────────────────────────────────────────

def draw_panel_bg(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),   (248, 285),  (15, 15, 15), -1)
    cv2.rectangle(overlay, (252, 0), (638, 145),  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


def draw_risk_badge(frame, risk_level: str):
    colours = {
        "LOW":      ( 20, 110,  20),
        "MEDIUM":   (  0, 120, 210),
        "HIGH":     (  0,  40, 190),
        "CRITICAL": (  0,   0, 175),
    }
    c = colours.get(risk_level, (20, 110, 20))
    cv2.rectangle(frame, (8, 150), (240, 176), c, -1)
    cv2.putText(frame, f"RISK: {risk_level}", (14, 169),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_left_panel(frame, blink_count: int, calibrating: bool, calib_pct: int):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "AI Interview Monitor",             (10,  26), f, 0.56, (0, 230, 0),   2)

    # Mic status — green when actively capturing speech, yellow when processing
    mic_colour = (0, 230, 210) if "Listening" not in listening_status else (0, 255, 80)
    if listening_status == "Processing":
        mic_colour = (0, 200, 255)
    cv2.putText(frame, f"Mic:        {listening_status}", (10,  58), f, 0.47, mic_colour,     1)

    cv2.putText(frame, f"Difficulty: {current_difficulty}",(10,  82), f, 0.47, (220, 210, 0), 1)

    score_str = str(current_score) if isinstance(current_score, int) else current_score
    cv2.putText(frame, f"Score:      {score_str}/10",      (10, 106), f, 0.47, (220, 210, 0), 1)

    draw_risk_badge(frame, detector.risk_level)

    cv2.putText(frame, f"Look away:  {int(look_away_duration)}s", (10, 202), f, 0.44, (200, 200, 200), 1)
    cv2.putText(frame, f"Incidents:  {len(detector.incidents)}",  (10, 224), f, 0.44, (200, 200, 200), 1)
    cv2.putText(frame, f"Blinks:     {blink_count}",              (10, 246), f, 0.44, (200, 200, 200), 1)

    if calibrating:
        bar_w = int(228 * calib_pct / 100)
        cv2.rectangle(frame, (10, 260), (238, 276), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 260), (10 + bar_w, 276), (0, 180, 220), -1)
        cv2.putText(frame, f"Calibrating... {calib_pct}%", (12, 273), f, 0.38, (255, 255, 255), 1)


def draw_question_panel(frame):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Question:", (258, 26), f, 0.56, (255, 255, 255), 2)
    for i, line in enumerate(wrap_question(current_question)):
        cv2.putText(frame, line, (258, 56 + i * 22), f, 0.44, (215, 215, 215), 1)


def draw_alert_banner(frame, text: str, y: int, colour=(0, 0, 160)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 26), (640, y + 8), colour, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)


# ── Main proctoring loop ──────────────────────────────────────────────────────

def proctoringAlgo():
    global look_away_start, look_away_duration
    global _last_objectName, _last_headPose
    global _window_focused, _tab_switched, _clipboard_used
    global _pose_calibrated, _calib_frame_count

    cam = open_camera()

    blinkCount  = 0
    frame_count = 0
    objectName  = []
    headPose    = "Center"
    eyeStatus   = "Center"
    mouthStatus = "Mouth Close"

    threading.Thread(target=start_interview_async,     daemon=True).start()
    threading.Thread(target=interview_process,          daemon=True).start()
    threading.Thread(target=_start_focus_listener,     daemon=True).start()
    threading.Thread(target=_start_clipboard_listener, daemon=True).start()

    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        frame_record = [datetime.now().strftime("%H:%M:%S.%f")]

        # ── Face detection ──────────────────────────────────────────────────
        try:
            faceCount, faces = detectFace(frame)
        except Exception:
            data_record.append(frame_record)
            continue

        if faces is None:
            faces = []
        valid_faces = [f for f in faces if is_valid_dlib_rect(f)]
        if faceCount == 1 and not valid_faces:
            faceCount = 0

        face_history.append(faceCount)
        if len(face_history) > 5:
            face_history.pop(0)
        avg = sum(face_history) / len(face_history)
        if   avg >= 1.5: faceCount = 2
        elif avg < 0.5:  faceCount = 0
        else:             faceCount = 1

        frame_record.append(faceCount_detection(faceCount))

        # ── Proctoring ──────────────────────────────────────────────────────
        if faceCount == 1 and valid_faces:
            sf = valid_faces

            # Blink (debounced — counts each blink event only once)
            try:
                bs = isBlinking(sf, frame)
                if isinstance(bs, tuple) and len(bs) >= 3:
                    if bs[2] == "Blink":
                        blinkCount += 1
                    frame_record.append(f"Blink: {bs[2]} ({blinkCount})")
                else:
                    frame_record.append("Blink: unknown")
            except Exception:
                frame_record.append("Blink: error")

            # Gaze
            try:
                eyeStatus = gazeDetection(sf, frame) or "Center"
                frame_record.append(f"Eye: {eyeStatus}")
            except Exception:
                eyeStatus = "Center"

            # Mouth (dynamic threshold — won't fire during normal speech)
            try:
                mouthStatus = mouthTrack(sf, frame) or "Mouth Close"
                frame_record.append(mouthStatus)
            except Exception:
                mouthStatus = "Mouth Close"

            # Objects (every 5 frames)
            if frame_count % 5 == 0:
                try:
                    objectName       = detectObject(frame)
                    _last_objectName = objectName
                except Exception:
                    objectName = _last_objectName
            else:
                objectName = _last_objectName
            frame_record.append(objectName)

            # Head pose (every 3 frames, with calibration)
            if frame_count % 3 == 0:
                try:
                    r        = head_pose_detection(sf, frame)
                    headPose = r if isinstance(r, str) else "Center"
                    _last_headPose = headPose
                    # Track calibration progress
                    if not _pose_calibrated:
                        _calib_frame_count = min(_calib_frame_count + 1, CALIB_TOTAL)
                        if _calib_frame_count >= CALIB_TOTAL:
                            _pose_calibrated = True
                except Exception:
                    headPose = _last_headPose
            else:
                headPose = _last_headPose
            frame_record.append(headPose)

            # Look-away timer
            looking_away = (
                headPose in ("Head Left", "Head Right")
                or eyeStatus.lower() in ("left", "right")
            )
            if looking_away:
                if look_away_start is None:
                    look_away_start = time.time()
                look_away_duration = time.time() - look_away_start
            else:
                look_away_start    = None
                look_away_duration = 0.0
        else:
            look_away_start    = None
            look_away_duration = 0.0

        # ── Cheating detector update ────────────────────────────────────────
        with _focus_lock:
            win_foc = _window_focused
            clip_u  = _clipboard_used
            _clipboard_used = False

        detector.update(
            face_count    = faceCount,
            head_pose     = headPose,
            eye_status    = eyeStatus,
            mouth_status  = mouthStatus,
            object_names  = objectName,
            blink_count   = blinkCount,
            window_focused= win_foc,
            clipboard_used= clip_u,
        )

        data_record.append(frame_record)

        # ── Dashboard ───────────────────────────────────────────────────────
        draw_panel_bg(frame)

        calib_pct = int(100 * _calib_frame_count / CALIB_TOTAL)
        draw_left_panel(frame, blinkCount, not _pose_calibrated, calib_pct)
        draw_question_panel(frame)

        # Alert banners stacked from bottom up
        banner_y = 462
        if detector.risk_level == "CRITICAL":
            draw_alert_banner(frame, "!! CRITICAL CHEATING RISK DETECTED !!",
                              banner_y, (0, 0, 170))
            banner_y -= 42

        if look_away_duration > LOOK_AWAY_LIMIT:
            draw_alert_banner(frame,
                              f"WARNING: Looking away ({int(look_away_duration)}s)",
                              banner_y, (0, 50, 170))
            banner_y -= 42

        if faceCount == 0:
            draw_alert_banner(frame, "NO FACE DETECTED", banner_y, (0, 30, 150))
            banner_y -= 42
        elif faceCount > 1:
            draw_alert_banner(frame, "MULTIPLE FACES IN FRAME", banner_y, (0, 30, 150))
            banner_y -= 42

        # Banned objects only — never "person" when candidate's face is in frame
        banned_labels = {
            "cell phone", "phone", "mobile phone", "book", "notebook",
            "laptop", "mouse", "keyboard", "remote", "scissors", "bottle"
        }
        banned_found = [lbl for lbl, _ in objectName if lbl.lower() in banned_labels]
        if banned_found:
            draw_alert_banner(frame, f"BANNED OBJECT: {', '.join(banned_found[:2])}",
                              banner_y, (0, 20, 130))

        # Calibration overlay
        if not _pose_calibrated:
            cv2.putText(frame, "Hold still — calibrating head pose...",
                        (258, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (0, 200, 220), 1)

        cv2.imshow("AI Interview System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nInterview ended.")
            break

    cam.release()
    cv2.destroyAllWindows()

    report = detector.final_report()
    print("\n" + report)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"integrity_report_{ts}.txt", "w") as f:
        f.write(report)
    print(f"Report saved: integrity_report_{ts}.txt")
    with open("activity.txt", "w") as f:
        f.write("\n".join(map(str, data_record)))
    print("Activity log saved: activity.txt")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    proctoringAlgo()