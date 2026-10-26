import re
import time
import queue
import threading
from collections import deque
import numpy as np

# ------------------------------------------------------------
# OPTIONAL MODULES
# ------------------------------------------------------------

# Windows native TTS is used instead of pyttsx3.
# This avoids the repeated-engine/audio playback problem.

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import winsound
except Exception:
    winsound = None


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1

MIN_THRESHOLD = 30
MAX_THRESHOLD = 1800
NOISE_MULT = 4.0

END_RATIO = 0.5
SILENCE_BLOCKS = 10

# Quiet microphones are boosted before recognition.
TARGET_PEAK = 20000
MAX_GAIN = 40.0

MAX_UTTERANCE_BLOCKS = 80
PUSH_TO_TALK_WAIT = 6.0

TTS_RATE = 175

# Set True to print a line every time speech is actually sent
# to Windows SAPI.
TTS_DEBUG = True

WAKE_PATTERN = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?(?:recall\s*cam|recall)\b[,:]?\s*(.*)$"
)


# ============================================================
# VOICE ASSISTANT
# ============================================================

class VoiceAssistant:

    def __init__(
        self,
        on_command,
        speak_enabled=True,
        mic_device=None
    ):
        self.on_command = on_command
        self.speak_enabled = speak_enabled
        self.mic_device = mic_device

        self.continuous = False
        self.listening = False

        # Speech queue
        self._speech_queue = queue.Queue()

        # True while Windows is speaking
        self._speaking = threading.Event()

        # Request one-shot listening
        self._listen_request = threading.Event()

        # Global shutdown
        self._stop = threading.Event()

        # Speech recognizer
        self.recognizer = (
            sr.Recognizer()
            if sr is not None
            else None
        )

        # ----------------------------------------------------
        # CAPABILITIES
        # ----------------------------------------------------

        # Windows native SAPI is available on Windows.
        # No pyttsx3 dependency is required.
        self.can_speak = True

        self.can_listen = (
            sd is not None
            and sr is not None
        )

        # ----------------------------------------------------
        # MICROPHONE RESOLUTION
        # ----------------------------------------------------

        if self.can_listen:
            try:
                sd.query_devices(
                    self.mic_device,
                    kind="input"
                )

            except Exception:

                working = self._probe_devices()

                if working:
                    print(
                        f"[VOICE] Mic {mic_device} unavailable, "
                        f"using {working[0][1]} "
                        f"(device {working[0][0]})"
                    )

                    self.mic_device = working[0][0]

                else:
                    print(
                        "[VOICE] No working microphone found."
                    )

        # ----------------------------------------------------
        # START TTS THREAD
        # ----------------------------------------------------

        if self.can_speak:
            threading.Thread(
                target=self._tts_loop,
                daemon=True
            ).start()

        # ----------------------------------------------------
        # START LISTENING THREAD
        # ----------------------------------------------------

        if self.can_listen:
            threading.Thread(
                target=self._listen_loop,
                daemon=True
            ).start()

    # ========================================================
    # STATUS
    # ========================================================

    def print_status(self):

        if self.can_speak:
            print(
                "[VOICE] Spoken answers: ready"
            )
        else:
            print(
                "[VOICE] Spoken answers unavailable. "
                "Windows SAPI is not available."
            )

        if self.can_listen:

            try:
                device = sd.query_devices(
                    self.mic_device,
                    kind="input"
                )["name"]

            except Exception:
                device = "unknown"

            print(
                f"[VOICE] Voice commands: ready, "
                f"microphone = {device}\n"
                "        (press V in the camera window, "
                "or type 'voice on'; test the mic with: "
                "python voice.py --level)"
            )

        else:

            print(
                "[VOICE] Voice commands unavailable. "
                "Run: pip install SpeechRecognition sounddevice"
            )

    def status_text(self):

        if self.listening:
            return "LISTENING"

        if self.continuous and self.can_listen:
            return "VOICE ON (say 'recall ...')"

        return ""

    # ========================================================
    # SPEAKING
    # ========================================================

    def say(self, text):

        if (
            not self.speak_enabled
            or not self.can_speak
            or not text
        ):
            return

        # Don't let a huge backlog build up.
        while self._speech_queue.qsize() > 3:

            try:
                self._speech_queue.get_nowait()

            except queue.Empty:
                break

        self._speech_queue.put(str(text))

    def mute(self):

        self.speak_enabled = False

        while True:

            try:
                self._speech_queue.get_nowait()

            except queue.Empty:
                break

    def _tts_loop(self):
        """
        Speak queued text using Windows native System.Speech/SAPI.

        Each queued sentence is processed sequentially.
        This replaces pyttsx3 completely.
        """

        import subprocess
        import os

        while not self._stop.is_set():

            try:
                text = self._speech_queue.get(
                    timeout=0.2
                )

            except queue.Empty:
                continue

            if not text:
                self._speech_queue.task_done()
                continue

            self._speaking.set()

            try:

                if TTS_DEBUG:
                    print(
                        f"[VOICE] speaking: {text[:80]}"
                    )

                # ------------------------------------------------
                # Put the actual text into an environment variable.
                #
                # This avoids PowerShell quote/escape problems.
                # ------------------------------------------------

                env = os.environ.copy()

                env["RECALLCAM_TTS_TEXT"] = str(text)

                # ------------------------------------------------
                # Native Windows SAPI
                # ------------------------------------------------

                powershell_code = r'''
Add-Type -AssemblyName System.Speech

$text = $env:RECALLCAM_TTS_TEXT

$s = New-Object System.Speech.Synthesis.SpeechSynthesizer

$s.Rate = 0
$s.Volume = 100

try {
    $s.Speak($text)
}
finally {
    $s.Dispose()
}
'''

                result = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        powershell_code
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                    creationflags=getattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                        0
                    ),
                    env=env
                )

                # ------------------------------------------------
                # TTS ERROR
                # ------------------------------------------------

                if result.returncode != 0:

                    error = result.stderr.strip()

                    print(
                        "[VOICE] Windows TTS error: "
                        f"{error or 'unknown error'}"
                    )

                elif TTS_DEBUG:

                    print(
                        "[VOICE] speech finished"
                    )

            except subprocess.TimeoutExpired:

                print(
                    "[VOICE] TTS timed out; "
                    "skipping this sentence."
                )

            except Exception as e:

                print(
                    f"[VOICE] Speech error: {e}"
                )

            finally:

                self._speech_queue.task_done()

                # Small gap before another utterance.
                time.sleep(0.15)

                # Only clear speaking when queue is empty.
                if self._speech_queue.empty():
                    self._speaking.clear()

    # ========================================================
    # MICROPHONE PROBING
    # ========================================================

    def _probe_devices(self):
        """
        Return a list of (index, name) for every input device
        that can actually be opened right now.
        """

        working = []

        if sd is None:
            return working

        block = int(
            SAMPLE_RATE * BLOCK_SECONDS
        )

        for index, device in enumerate(
            sd.query_devices()
        ):

            if device["max_input_channels"] < 1:
                continue

            try:

                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=block,
                    device=index
                ):
                    pass

                working.append(
                    (
                        index,
                        device["name"]
                    )
                )

            except Exception:
                continue

        return working

    # ========================================================
    # LISTENING
    # ========================================================

    def listen_once(self):

        if not self.can_listen:

            print(
                "[VOICE] Voice commands unavailable. "
                "Run: pip install SpeechRecognition sounddevice"
            )

            return

        self._listen_request.set()

    def set_continuous(self, enabled):

        if enabled and not self.can_listen:

            print(
                "[VOICE] Voice commands unavailable. "
                "Run: pip install SpeechRecognition sounddevice"
            )

            return False

        self.continuous = enabled

        return True

    def stop(self):

        self._stop.set()

    # ========================================================
    # AUDIO LEVEL
    # ========================================================

    @staticmethod
    def _rms(chunk):

        return float(
            np.sqrt(
                np.mean(
                    chunk.astype(np.float32) ** 2
                )
            )
        )

    def _beep(self):
        """
        Short ready tone so you know when to start talking.
        """

        if winsound is None:
            return

        try:
            winsound.Beep(
                900,
                100
            )

        except Exception:
            pass

    # ========================================================
    # RECORD
    # ========================================================

    def _record(
        self,
        start_timeout,
        cue=False
    ):
        """
        Record one utterance.

        start_timeout=None waits forever for speech to begin.

        Returns:
            (raw PCM bytes or None, info dict)

        info["reason"]:
            ok
            timeout
            speaking
            error
        """

        info = {
            "reason": "ok",
            "noise": 0.0,
            "threshold": 0.0,
            "end_threshold": 0.0,
            "peak": 0.0,
            "error": ""
        }

        block = int(
            SAMPLE_RATE * BLOCK_SECONDS
        )

        preroll = deque(maxlen=3)

        frames = []

        started = False
        silent_blocks = 0

        try:

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=block,
                device=self.mic_device
            ) as stream:

                # ------------------------------------------------
                # Measure background noise
                # ------------------------------------------------

                noise = []

                for _ in range(3):

                    data, _ = stream.read(block)

                    noise.append(
                        self._rms(data[:, 0])
                    )

                info["noise"] = float(
                    np.mean(noise)
                )

                info["threshold"] = float(
                    min(
                        MAX_THRESHOLD,
                        max(
                            MIN_THRESHOLD,
                            info["noise"] * NOISE_MULT
                        )
                    )
                )

                # ------------------------------------------------
                # Hysteresis
                # ------------------------------------------------

                info["end_threshold"] = float(
                    max(
                        info["noise"] * 2.0,
                        info["threshold"] * END_RATIO
                    )
                )

                # ------------------------------------------------
                # Beep
                # ------------------------------------------------

                if cue:

                    self._beep()

                    # Throw away audio captured while beep played.
                    try:

                        waiting = stream.read_available

                        if waiting:
                            stream.read(waiting)

                    except Exception:
                        pass

                self.listening = True

                began_waiting = time.time()

                # ------------------------------------------------
                # RECORD LOOP
                # ------------------------------------------------

                while not self._stop.is_set():

                    # Never transcribe our own voice.
                    if self._speaking.is_set():

                        info["reason"] = "speaking"

                        return None, info

                    data, _ = stream.read(block)

                    chunk = data[:, 0].copy()

                    level = self._rms(chunk)

                    info["peak"] = max(
                        info["peak"],
                        level
                    )

                    # ------------------------------------------------
                    # WAIT FOR SPEECH
                    # ------------------------------------------------

                    if not started:

                        if level > info["threshold"]:

                            started = True

                            frames.extend(preroll)

                            frames.append(chunk)

                        else:

                            preroll.append(chunk)

                            # One-shot timeout
                            if (
                                start_timeout is not None
                                and
                                time.time() - began_waiting
                                > start_timeout
                            ):

                                info["reason"] = "timeout"

                                return None, info

                            # Continuous mode switched off.
                            if (
                                start_timeout is None
                                and
                                not self.continuous
                            ):

                                info["reason"] = "timeout"

                                return None, info

                    # ------------------------------------------------
                    # RECORDING SPEECH
                    # ------------------------------------------------

                    else:

                        frames.append(chunk)

                        if (
                            level
                            < info["end_threshold"]
                        ):

                            silent_blocks += 1

                        else:

                            silent_blocks = 0

                        # ------------------------------------------------
                        # END SPEECH
                        # ------------------------------------------------

                        if (
                            silent_blocks
                            >= SILENCE_BLOCKS
                            or
                            len(frames)
                            >= MAX_UTTERANCE_BLOCKS
                        ):

                            break

        except Exception as e:

            info["reason"] = "error"

            info["error"] = str(e)

            return None, info

        finally:

            self.listening = False

        if not frames:

            info["reason"] = "timeout"

            return None, info

        return (
            np.concatenate(frames).tobytes(),
            info
        )

    # ========================================================
    # EXPLAIN FAILURE
    # ========================================================

    def _explain_failure(self, info):

        reason = info["reason"]

        if reason == "speaking":

            print(
                "[VOICE] I was still talking. "
                "Press V again."
            )

        elif reason == "error":

            print(
                f"[VOICE] Microphone error: "
                f"{info['error']}"
            )

            print(
                "        Check Windows Settings > "
                "Privacy > Microphone, "
                "or list devices with: "
                "python voice.py --mics"
            )

        else:

            print(
                "[VOICE] I didn't hear anything "
                f"(mic peaked at {info['peak']:.0f}, "
                f"needed {info['threshold']:.0f})."
            )

            if info["peak"] < 30:

                print(
                    "        The mic level is almost zero: "
                    "it may be muted, blocked by Windows "
                    "privacy settings, or the wrong device.\n"
                    "        Run: python voice.py --mics "
                    "then set MIC_DEVICE in main.py."
                )

            else:

                print(
                    "        Speak after the beep. "
                    "If your voice stays below the threshold, "
                    "run: python voice.py --level"
                )

    # ========================================================
    # NORMALIZE AUDIO
    # ========================================================

    @staticmethod
    def _normalize(raw_audio):
        """
        Boost quiet recordings so the recognizer
        can understand them.
        """

        samples = np.frombuffer(
            raw_audio,
            dtype=np.int16
        ).astype(np.float32)

        if samples.size == 0:
            return raw_audio

        peak = float(
            np.max(
                np.abs(samples)
            )
        )

        if peak < 1.0:
            return raw_audio

        gain = min(
            MAX_GAIN,
            TARGET_PEAK / peak
        )

        if gain <= 1.0:
            return raw_audio

        boosted = np.clip(
            samples * gain,
            -32768,
            32767
        )

        return boosted.astype(
            np.int16
        ).tobytes()

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    def _transcribe(self, raw_audio):

        try:

            raw_audio = self._normalize(
                raw_audio
            )

            audio = sr.AudioData(
                raw_audio,
                SAMPLE_RATE,
                2
            )

            return self.recognizer.recognize_google(
                audio
            )

        except sr.UnknownValueError:

            return None

        except sr.RequestError as e:

            print(
                "[VOICE] Speech service unavailable "
                f"(needs internet): {e}"
            )

            return None

        except Exception as e:

            print(
                f"[VOICE] Recognition error: {e}"
            )

            return None

    # ========================================================
    # LISTEN LOOP
    # ========================================================

    def _listen_loop(self):

        while not self._stop.is_set():

            one_shot = (
                self._listen_request.is_set()
            )

            # ------------------------------------------------
            # Nothing to do
            # ------------------------------------------------

            if (
                not one_shot
                and
                not self.continuous
            ):

                self._listen_request.wait(
                    0.2
                )

                continue

            self._listen_request.clear()

            # ------------------------------------------------
            # Wait until RecallCam finishes speaking.
            # ------------------------------------------------

            while (
                self._speaking.is_set()
                and
                not self._stop.is_set()
            ):

                time.sleep(0.05)

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            raw, info = self._record(
                PUSH_TO_TALK_WAIT
                if one_shot
                else None,
                cue=one_shot
            )

            # ------------------------------------------------
            # Recording failed
            # ------------------------------------------------

            if raw is None:

                if (
                    one_shot
                    or
                    info["reason"] == "error"
                ):

                    self._explain_failure(
                        info
                    )

                if info["reason"] == "error":

                    time.sleep(2.0)

                continue

            # ------------------------------------------------
            # Speech recognition
            # ------------------------------------------------

            text = self._transcribe(
                raw
            )

            if not text:

                if one_shot:

                    print(
                        "[VOICE] Sorry, "
                        "I couldn't understand that."
                    )

                    self.say(
                        "Sorry, I didn't catch that."
                    )

                continue

            # ------------------------------------------------
            # Normalize recognized text
            # ------------------------------------------------

            text = text.lower().strip()

            print(
                f"\n[VOICE] Heard: {text}"
            )

            # ------------------------------------------------
            # Process command
            # ------------------------------------------------

            try:

                # One-shot mode:
                # send everything directly to RecallCam.
                if one_shot:

                    self.on_command(text)

                    continue

                # ------------------------------------------------
                # Continuous mode:
                # only respond to wake word.
                # ------------------------------------------------

                match = WAKE_PATTERN.match(
                    text
                )

                if not match:
                    continue

                command = (
                    match.group(1)
                    .strip()
                )

                if command:

                    self.on_command(
                        command
                    )

                else:

                    self.say(
                        "Yes?"
                    )

                    self._listen_request.set()

            except Exception as e:

                print(
                    f"[VOICE] Command error: {e}"
                )


# ============================================================
# MICROPHONE LIST
# ============================================================

def list_microphones():

    if sd is None:

        print(
            "sounddevice is not installed: "
            "pip install sounddevice"
        )

        return

    print(
        "Input devices "
        "(use the number as MIC_DEVICE in main.py):\n"
    )

    default_in = None

    try:

        default_in = sd.default.device[0]

    except Exception:
        pass

    for index, device in enumerate(
        sd.query_devices()
    ):

        if device["max_input_channels"] > 0:

            mark = (
                "  <- default"
                if index == default_in
                else ""
            )

            print(
                f"  {index}: "
                f"{device['name']}{mark}"
            )


# ============================================================
# MICROPHONE LEVEL METER
# ============================================================

def level_meter(
    seconds=8,
    device=None
):
    """
    Live microphone level + whether RecallCam
    would trigger on it.
    """

    if sd is None:

        print(
            "sounddevice is not installed: "
            "pip install sounddevice"
        )

        return

    try:

        name = sd.query_devices(
            device,
            kind="input"
        )["name"]

    except Exception:

        name = "default"

    print(
        f"Microphone: {name}"
    )

    print(
        f"Stay QUIET for the first second, "
        f"then speak normally for {seconds} seconds.\n"
    )

    levels = []

    block = int(
        SAMPLE_RATE * BLOCK_SECONDS
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=block,
        device=device
    ) as stream:

        end = (
            time.time()
            + seconds
        )

        while time.time() < end:

            data, _ = stream.read(
                block
            )

            level = float(
                np.sqrt(
                    np.mean(
                        data[:, 0]
                        .astype(np.float32) ** 2
                    )
                )
            )

            levels.append(level)

            bar = "#" * min(
                60,
                int(level / 25)
            )

            print(
                f"{level:7.0f} {bar}"
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    noise = (
        float(np.mean(levels[:3]))
        if levels
        else 0.0
    )

    start = min(
        MAX_THRESHOLD,
        max(
            MIN_THRESHOLD,
            noise * NOISE_MULT
        )
    )

    peak = (
        max(levels)
        if levels
        else 0.0
    )

    print(
        f"\nBackground noise : {noise:.0f}"
    )

    print(
        f"Start threshold  : {start:.0f}"
    )

    print(
        f"Your peak level  : {peak:.0f}"
    )

    if peak < 10:

        print(
            "\nRESULT: almost silent. "
            "Wrong device, muted mic, or "
            "Windows privacy settings are blocking it."
        )

    elif peak < start:

        print(
            "\nRESULT: too quiet to trigger. "
            "Try another device, speak closer, "
            "or raise Windows Sound settings > "
            "Input volume."
        )

    elif peak < start * 2:

        print(
            "\nRESULT: works, but the mic is quiet. "
            "RecallCam boosts the audio automatically; "
            "a louder device will recognise better."
        )

    else:

        print(
            "\nRESULT: mic level is good."
        )


# ============================================================
# COMMAND-LINE DEVICE ARGUMENT
# ============================================================

def _device_arg(flag):
    """
    python voice.py --level 29
        -> 29
    """

    import sys

    index = sys.argv.index(flag)

    if index + 1 < len(sys.argv):

        value = sys.argv[index + 1]

        if value.lstrip("-").isdigit():

            return int(value)

    return None


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    import sys

    # --------------------------------------------------------
    # List microphones
    # --------------------------------------------------------

    if "--mics" in sys.argv:

        list_microphones()

        sys.exit(0)

    # --------------------------------------------------------
    # Microphone level
    # --------------------------------------------------------

    if "--level" in sys.argv:

        level_meter(
            device=_device_arg("--level")
        )

        sys.exit(0)

    # --------------------------------------------------------
    # Voice self-test
    # --------------------------------------------------------

    assistant = None

    def handle_command(text):

        print(
            f"Command: {text}"
        )

        assistant.say(
            f"You said {text}"
        )

    assistant = VoiceAssistant(
        on_command=handle_command
    )

    assistant.print_status()

    # --------------------------------------------------------
    # TTS test
    # --------------------------------------------------------

    assistant.say(
        "Voice test. "
        "Say something after the beep."
    )

    time.sleep(3)

    # --------------------------------------------------------
    # Microphone test
    # --------------------------------------------------------

    assistant.listen_once()

    # Give the listening thread time to finish.
    time.sleep(12)

    assistant.stop()