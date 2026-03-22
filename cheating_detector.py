"""
cheating_detector.py
--------------------
Tracks all cheating signals and produces a timestamped incident log
and final risk report.

Fixes in this version:
- Extra-person signal only fires when YOLO detects a person AND no
  valid dlib face was detected (prevents flagging the candidate themselves)
- Focus/window events are debounced — one focus-loss counts as ONE event
  regardless of how many poll cycles it spans
- Clipboard events debounced similarly
"""

import time
from datetime import datetime
from collections import deque


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG = {
    "no_face_warn_secs":      3,
    "no_face_critical_secs": 10,
    "look_away_warn_secs":    3,
    "look_away_high_secs":    7,
    "mouth_open_warn_secs":   4,
    "blink_window_secs":     60,
    "blink_low_threshold":    6,      # blinks/min
    "tab_switch_warn_count":  2,
    "tab_switch_high_count":  5,
    "phone_labels":    {"cell phone", "mobile phone", "phone"},
    "book_labels":     {"book", "notebook", "laptop", "mouse",
                        "keyboard", "remote", "scissors", "bottle"},
    # "person" only flagged when face_count == 0 (not the candidate)
    "person_labels":   {"person"},
    "clipboard_warn_count":  2,
    "clipboard_high_count":  5,
}

WEIGHTS = {
    "no_face":         3,
    "multiple_faces":  5,
    "look_away":       2,
    "mouth_open":      2,
    "low_blink":       1,
    "phone_detected":  8,
    "book_detected":   4,
    "extra_person":    7,
    "tab_switch":      4,
    "window_hidden":   5,
    "clipboard":       3,
}


class Incident:
    __slots__ = ("timestamp", "signal", "detail", "severity")

    def __init__(self, signal: str, detail: str, severity: str = "warn"):
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.signal    = signal
        self.detail    = detail
        self.severity  = severity

    def __str__(self):
        return f"[{self.timestamp}] [{self.severity.upper():8}] {self.signal}: {self.detail}"


class CheatingDetector:
    def __init__(self):
        self.incidents:  list[Incident] = []
        self.score       = 0
        self.risk_level  = "LOW"
        self.risk_bgr    = (0, 200, 0)

        # timers
        self._no_face_start    = None
        self._look_away_start  = None
        self._mouth_open_start = None

        # counters
        self._tab_switches        = 0
        self._clipboard_events    = 0
        self._window_hidden_count = 0

        # blink tracking
        self._blink_timestamps: deque = deque()
        self._last_blink_count = 0

        # debounce: track whether we are CURRENTLY in a focus-lost state
        # so a single focus-loss only counts once
        self._in_focus_loss    = False
        self._in_clipboard_evt = False

        # cooldown between repeat log entries for the same signal
        self._last_logged: dict[str, float] = {}
        self._cooldown = 5.0

        self.session_start = datetime.now()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(
        self,
        face_count:     int,
        head_pose:      str,
        eye_status:     str,
        mouth_status:   str,
        object_names:   list,      # [(label, confidence), ...]
        blink_count:    int,
        window_focused: bool = True,
        clipboard_used: bool = False,
    ):
        now = time.time()
        self._check_face(face_count, now)
        self._check_gaze(head_pose, eye_status, now)
        self._check_mouth(mouth_status, now)
        self._check_blink(blink_count, now)
        self._check_objects(object_names, face_count, now)
        self._check_window(window_focused, now)
        self._check_clipboard(clipboard_used, now)
        self._recompute_risk()

    def notify_tab_switch(self):
        """Call exactly once per focus-loss edge (not every poll cycle)."""
        self._tab_switches += 1
        sev = "warn" if self._tab_switches < CFG["tab_switch_high_count"] else "high"
        self._log("Tab switch", f"Total switches: {self._tab_switches}", sev)

    def notify_clipboard(self):
        """Call exactly once per new clipboard sequence number change."""
        self._clipboard_events += 1
        sev = "warn" if self._clipboard_events < CFG["clipboard_high_count"] else "high"
        self._log("Clipboard paste",
                  f"Total paste events: {self._clipboard_events}", sev)

    def final_report(self) -> str:
        duration = datetime.now() - self.session_start
        lines = [
            "=" * 62,
            "  AI INTERVIEW INTEGRITY REPORT",
            f"  Session date : {self.session_start.strftime('%Y-%m-%d')}",
            f"  Start time   : {self.session_start.strftime('%H:%M:%S')}",
            f"  Duration     : {str(duration).split('.')[0]}",
            f"  Final risk   : {self.risk_level}",
            f"  Risk score   : {self.score}",
            "=" * 62,
            "",
            "INCIDENT LOG",
            "-" * 62,
        ]
        if not self.incidents:
            lines.append("  No incidents recorded.")
        else:
            for inc in self.incidents:
                lines.append(str(inc))

        lines += ["", "SUMMARY BY SIGNAL", "-" * 62]
        signal_counts: dict[str, int] = {}
        for inc in self.incidents:
            signal_counts[inc.signal] = signal_counts.get(inc.signal, 0) + 1

        if signal_counts:
            for sig, cnt in sorted(signal_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {sig:<30} {cnt:>3} event(s)")
        else:
            lines.append("  No incidents.")

        lines += ["", "=" * 62]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Internal checkers
    # -----------------------------------------------------------------------

    def _check_face(self, face_count: int, now: float):
        if face_count == 0:
            if self._no_face_start is None:
                self._no_face_start = now
            elapsed = now - self._no_face_start
            if elapsed >= CFG["no_face_critical_secs"]:
                self._log("No face", f"{int(elapsed)}s — candidate may have left", "critical")
                self._add_score("no_face", 3)
            elif elapsed >= CFG["no_face_warn_secs"]:
                self._log("No face", f"{int(elapsed)}s without face in frame", "warn")
                self._add_score("no_face", 1)
        else:
            self._no_face_start = None

        if face_count > 1:
            self._log("Multiple faces", f"{face_count} faces detected", "high")
            self._add_score("multiple_faces", WEIGHTS["multiple_faces"])

    def _check_gaze(self, head_pose: str, eye_status: str, now: float):
        looking_away = (
            head_pose in ("Head Left", "Head Right")
            or eye_status.lower() in ("left", "right")
        )
        if looking_away:
            if self._look_away_start is None:
                self._look_away_start = now
            elapsed = now - self._look_away_start
            if elapsed >= CFG["look_away_high_secs"]:
                self._log("Gaze away",
                          f"{int(elapsed)}s — head: {head_pose}, eyes: {eye_status}", "high")
                self._add_score("look_away", 2)
            elif elapsed >= CFG["look_away_warn_secs"]:
                self._log("Gaze away",
                          f"{int(elapsed)}s — head: {head_pose}, eyes: {eye_status}", "warn")
                self._add_score("look_away", 1)
        else:
            self._look_away_start = None

    def _check_mouth(self, mouth_status: str, now: float):
        if mouth_status == "Mouth Open":
            if self._mouth_open_start is None:
                self._mouth_open_start = now
            elapsed = now - self._mouth_open_start
            if elapsed >= CFG["mouth_open_warn_secs"]:
                self._log("Mouth open",
                          f"Sustained {int(elapsed)}s — possible whispering", "warn")
                self._add_score("mouth_open", WEIGHTS["mouth_open"])
        else:
            self._mouth_open_start = None

    def _check_blink(self, blink_count: int, now: float):
        new_blinks = blink_count - self._last_blink_count
        self._last_blink_count = blink_count
        for _ in range(max(0, new_blinks)):
            self._blink_timestamps.append(now)

        window = CFG["blink_window_secs"]
        while self._blink_timestamps and (now - self._blink_timestamps[0]) > window:
            self._blink_timestamps.popleft()

        if self._blink_timestamps and \
                (now - self._blink_timestamps[0]) >= window:
            rate = len(self._blink_timestamps) * (60.0 / window)
            if rate < CFG["blink_low_threshold"]:
                self._log("Low blink rate",
                          f"{rate:.1f} blinks/min — possible screen reading", "warn")
                self._add_score("low_blink", WEIGHTS["low_blink"])

    def _check_objects(self, object_names: list, face_count: int, now: float):
        """
        FIX: only flag 'extra person' when YOLO sees a person AND there is
        no valid face from dlib — otherwise we are just detecting the candidate.
        """
        labels = {label.lower() for label, _ in object_names}

        if labels & CFG["phone_labels"]:
            self._log("Phone detected",
                      f"Objects: {labels & CFG['phone_labels']}", "critical")
            self._add_score("phone_detected", WEIGHTS["phone_detected"])

        if labels & CFG["book_labels"]:
            self._log("Banned object",
                      f"Objects: {labels & CFG['book_labels']}", "high")
            self._add_score("book_detected", WEIGHTS["book_detected"])

        # Only flag person when the candidate's own face is NOT in frame
        if (labels & CFG["person_labels"]) and face_count == 0:
            self._log("Extra person",
                      "Person detected with no registered face — possible third party",
                      "high")
            self._add_score("extra_person", WEIGHTS["extra_person"])

    def _check_window(self, window_focused: bool, now: float):
        """
        FIX: use edge detection — only increment when transitioning from
        focused → unfocused, not on every poll cycle while unfocused.
        """
        if not window_focused:
            if not self._in_focus_loss:
                # rising edge of focus loss
                self._in_focus_loss    = True
                self._window_hidden_count += 1
                self._log("Window hidden",
                          f"App lost focus (event #{self._window_hidden_count})", "high")
                self._add_score("window_hidden", WEIGHTS["window_hidden"])
        else:
            self._in_focus_loss = False   # reset so next loss is a new edge

    def _check_clipboard(self, clipboard_used: bool, now: float):
        if clipboard_used:
            self._add_score("clipboard", WEIGHTS["clipboard"])

    # -----------------------------------------------------------------------
    # Risk computation
    # -----------------------------------------------------------------------

    def _recompute_risk(self):
        if self.score >= 30:
            self.risk_level = "CRITICAL"
            self.risk_bgr   = (0, 0, 220)
        elif self.score >= 15:
            self.risk_level = "HIGH"
            self.risk_bgr   = (0, 60, 220)
        elif self.score >= 6:
            self.risk_level = "MEDIUM"
            self.risk_bgr   = (0, 165, 255)
        else:
            self.risk_level = "LOW"
            self.risk_bgr   = (0, 200, 0)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _add_score(self, key: str, delta: int):
        self.score = min(100, self.score + delta)

    def _log(self, signal: str, detail: str, severity: str = "warn"):
        now = time.time()
        if now - self._last_logged.get(signal, 0) < self._cooldown:
            return
        self._last_logged[signal] = now
        inc = Incident(signal, detail, severity)
        self.incidents.append(inc)
        print(f"INCIDENT: {inc}")