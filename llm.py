import os
import re
from google import genai


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Change this without touching the code:
#
# PowerShell:
# $env:GEMINI_MODEL="gemini-3.8-flash"
#
# If you have another currently available Gemini model,
# you can change it here.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ------------------------------------------------------------
# GEMINI CLIENT
# ------------------------------------------------------------

class RecallLLM:

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        self.client = None
        self.enabled = False

        if not self.api_key:
            print(
                "[LLM] GEMINI_API_KEY not found. "
                "LLM fallback disabled."
            )
            return

        try:
            self.client = genai.Client(
                api_key=self.api_key
            )

            self.enabled = True

            print(
                f"[LLM] Gemini fallback enabled "
                f"({GEMINI_MODEL})"
            )

        except Exception as e:
            print(f"[LLM] Could not initialize Gemini: {e}")

    # --------------------------------------------------------
    # SHOULD USE LLM?
    # --------------------------------------------------------

    @staticmethod
    def looks_complex(command):
        """
        Decide locally whether a command is worth sending
        to Gemini.

        This is important because we DON'T want every unknown
        sentence to consume Gemini quota.
        """

        text = str(command).lower().strip()

        if not text:
            return False

        # Very short commands are normally simple commands.
        words = text.split()

        if len(words) <= 3:
            return False

        # Questions that clearly need natural-language reasoning.
        complex_patterns = [
            r"\bremember\b",
            r"\bthink\b",
            r"\bwhat could\b",
            r"\bwhich\b",
            r"\bwhy\b",
            r"\bhow\b",
            r"\bafter\b",
            r"\bbefore\b",
            r"\bthen\b",
            r"\bwhen i\b",
            r"\bwhen did i\b",
            r"\bwhere did i\b",
            r"\bwhat happened to\b",
            r"\bwhat happened with\b",
            r"\bnear\b",
            r"\bnext to\b",
            r"\bbeside\b",
            r"\baround\b",
            r"\byesterday\b",
            r"\btoday\b",
            r"\bearlier\b",
            r"\blater\b",
            r"\bago\b",
            r"\bput\b",
            r"\bleft\b",
            r"\bmoved\b",
            r"\bpicked\b",
            r"\bheld\b",
        ]

        for pattern in complex_patterns:
            if re.search(pattern, text):
                return True

        # Longer natural-language questions are also candidates.
        if len(words) >= 9:
            return True

        return False

    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

    def ask(self, question, memory_context):

        if not self.enabled:
            return None

        question = str(question).strip()

        if not question:
            return None

        prompt = f"""
You are RecallCam, a personal object-memory assistant.

Your job is to answer questions using ONLY the RecallCam
memory information provided below.

Do not invent objects, locations, times, events, or actions.

If the memory does not contain enough information, say that
you don't have enough information.

Keep the answer short and natural because the answer may be
spoken aloud by text-to-speech.

The user said:
"{question}"

RecallCam memory:
{memory_context}

Answer the user's question directly.
"""

        try:

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            answer = getattr(response, "text", None)

            if not answer:
                return None

            answer = answer.strip()

            # Remove accidental markdown formatting.
            answer = re.sub(
                r"^```(?:text)?\s*",
                "",
                answer,
                flags=re.IGNORECASE
            )

            answer = re.sub(
                r"\s*```$",
                "",
                answer
            )

            return answer.strip()

        except Exception as e:

            error = str(e)

            print(f"[LLM] Gemini error: {error}")

            return None