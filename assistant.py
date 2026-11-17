"""
Campus FAQ Assistant - AIP203 Lab 10
Single-file assistant combining Naive Bayes, optional Gemini API,
CLI chat loop, unit tests, and Streamlit web UI.

Run CLI:       python assistant.py
Run Web UI:    streamlit run assistant.py
Run Tests:     python -m unittest assistant.py
"""

import math
import os
import re
import unittest
from collections import Counter, defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Margin needed between best and second best score to accept an answer
MIN_MARGIN = 0.5
MIN_KNOWN_TOKENS = 1

# Filler words to ignore (keeping question words like 'where', 'when', 'who'
# because they help tell location/time/person queries apart)
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "be", "do", "does", "to", "of",
    "in", "on", "at", "it", "for", "and", "or", "how", "what", "can",
    "there", "this", "that", "please", "s",
}

# Intents with training examples and replies
INTENTS = {
    "greeting": {
        "examples": [
            "hi", "hello", "hey", "good morning", "good evening", "hi there",
        ],
        "reply": "Hey! Ask me anything about library hours, fees, exams, campus locations, or advisors.",
    },
    "goodbye": {
        "examples": [
            "bye", "goodbye", "see you later", "see you", "thanks bye",
        ],
        "reply": "Goodbye and good luck with your classes!",
    },
    "library_hours": {
        "examples": [
            "when is the library open", "library timings", "what time does the library close",
            "is the library open today", "library opening hours", "library hours on sunday",
        ],
        "reply": "The central library is open from 8:30 AM to 9:00 PM on weekdays, and 10:00 AM to 5:00 PM on weekends.",
    },
    "fees": {
        "examples": [
            "how do i pay my fees", "fee payment deadline", "where can i pay the hostel fee",
            "fees due date", "how to pay tuition", "pay hostel fee online",
        ],
        "reply": "Fees can be paid online via the student ERP portal (under Accounts & Fees) before the semester due date.",
    },
    "exam_schedule": {
        "examples": [
            "when are the exams", "exam dates", "when does the semester exam start",
            "exam timetable", "midterm exam schedule", "when is the exam for block 3",
        ],
        "reply": "Midterms start around mid-October, and end-sem exams begin in the first week of December. Timetables are posted on the notice board 2 weeks prior.",
    },
    "location": {
        "examples": [
            "where is the canteen", "how do i reach the admin office", "where is the placement cell",
            "location of block 2", "where is the library building", "find the admin block",
        ],
        "reply": "The canteen is next to Block B. The Admin Office is on the ground floor of Block A (Room 102).",
    },
    "advisor": {
        "examples": [
            "who is my faculty advisor", "how do i contact my mentor",
            "faculty advisor name", "my advisor email",
        ],
        "reply": "Your faculty advisor's name and contact email are listed on your student ERP profile under Academic Details.",
    },
}

FALLBACK = "Sorry, I didn't catch that. You can ask me about library hours, fees, exams, locations, or advisors (or type 'topics')."

TOPICS_LIST = (
    "You can ask me about:\n"
    "  1. Library Hours   - e.g. 'When is the library open?'\n"
    "  2. Fees            - e.g. 'How do I pay fees?'\n"
    "  3. Exam Schedule   - e.g. 'When do semester exams start?'\n"
    "  4. Campus Location - e.g. 'Where is the canteen?'\n"
    "  5. Faculty Advisor - e.g. 'Who is my faculty advisor?'"
)


def normalize(word):
    # Quick plural cleanup (e.g. fees -> fee, exams -> exam)
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [normalize(w) for w in words if w not in STOPWORDS]


class IntentClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, intents):
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()

        # Count words per intent
        for intent_name, data in intents.items():
            for example in data["examples"]:
                self.class_counts[intent_name] += 1
                for w in tokenize(example):
                    self.word_counts[intent_name][w] += 1
                    self.vocab.add(w)

        self.total_docs = sum(self.class_counts.values())
        self.class_totals = {c: sum(self.word_counts[c].values()) for c in self.class_counts}

    def scores(self, tokens):
        # Calculate log prior + log likelihood for each class
        vocab_size = len(self.vocab)
        result = {}

        for c, count in self.class_counts.items():
            # Prior P(c)
            score = math.log(count / self.total_docs)
            # Likelihood P(w|c) with +1 smoothing
            for w in tokens:
                word_freq = self.word_counts[c][w]
                score += math.log((word_freq + 1) / (self.class_totals[c] + vocab_size))
            result[c] = score

        return result

    def classify(self, text):
        tokens = [w for w in tokenize(text) if w in self.vocab]
        if len(tokens) < MIN_KNOWN_TOKENS:
            return None

        # Do not classify pure numbers or math expressions as campus FAQ intents
        if not any(not w.isdigit() for w in tokens):
            return None

        ranked = sorted(self.scores(tokens).items(), key=lambda x: x[1], reverse=True)
        (best_intent, best_score), (_, runner_up_score) = ranked[0], ranked[1]

        # Only return if we have a clear margin over the runner-up
        if best_score - runner_up_score < MIN_MARGIN:
            return None

        return best_intent


def ask_gemini(text, api_key=None):
    """Use Gemini API for general questions beyond campus FAQ templates."""
    key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not key or not str(key).strip():
        return None

    try:
        from google import genai
        client = genai.Client(api_key=key.strip())
        campus_facts = "\n".join(f"- {name}: {info['reply']}" for name, info in INTENTS.items())
        system_instruction = (
            "You are a friendly campus virtual assistant for university students. "
            "Answer clearly and concisely. "
            f"Here is verified campus information to reference when relevant:\n{campus_facts}"
        )

        # Prioritize lightweight and fast models that avoid quota overload
        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
        last_error = None

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=text,
                    config={
                        "system_instruction": system_instruction,
                        "http_options": {"timeout": 15000},
                    },
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as err:
                last_error = err
                err_str = str(err).lower()
                # If model is overloaded, rate-limited, unavailable, or not found on this API tier, try next candidate
                if any(w in err_str for w in ("demand", "overload", "503", "429", "exhausted", "unavailable", "404", "not_found", "not found")):
                    continue
                break

        if last_error:
            err_msg = str(last_error)
            if any(w in err_msg.lower() for w in ("demand", "overload", "503", "429")):
                return (
                    "The AI service is currently experiencing high demand. "
                    "Please try your question again in a moment, or ask about our core campus topics "
                    "(library hours, fees, exams, locations, or advisors)."
                )
            return f"[Gemini API Error: {last_error}]"

        return None
    except Exception as e:
        return f"[Gemini API Error: {e}]"


def respond(classifier, text, api_key=None):
    cleaned = text.strip().lower()
    if cleaned in {"topics", "help"}:
        return TOPICS_LIST

    # 1. Check local Naive Bayes classifier first
    intent = classifier.classify(text)
    if intent and intent in INTENTS:
        return INTENTS[intent]["reply"]

    # 2. If intent is unrecognized, consult Gemini API if key is available
    gemini_reply = ask_gemini(text, api_key=api_key)
    if gemini_reply:
        return gemini_reply

    # 3. Fallback when neither is available
    return FALLBACK


def main():
    classifier = IntentClassifier(INTENTS)
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    print("--- Campus Virtual Assistant (type 'quit' to exit) ---")
    if has_gemini:
        print("[Gemini API: Enabled]")
    else:
        print("[Gemini API: Disabled (set GEMINI_API_KEY environment variable to enable)]")
    print(TOPICS_LIST)
    print("-" * 50)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("Assistant:", INTENTS["goodbye"]["reply"])
            break

        print("Assistant:", respond(classifier, text))


def run_streamlit():
    """Minimal Streamlit web interface."""
    import streamlit as st

    st.title("Campus Virtual Assistant")
    st.write("Ask me about library hours, fees, exams, campus locations, or general queries.")

    with st.sidebar:
        st.subheader("AI Settings")
        default_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Gemini API Key (optional)",
            value=default_key,
            type="password",
            help="Enter your Google Gemini API key to enable AI answers for queries outside the standard campus FAQs.",
        )
        if api_key:
            st.success("Gemini API enabled")
        else:
            st.info("Using local Naive Bayes classifier")

    if "clf" not in st.session_state:
        st.session_state.clf = IntentClassifier(INTENTS)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                bot_reply = respond(st.session_state.clf, user_input, api_key=api_key)
            st.write(bot_reply)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.clf = IntentClassifier(INTENTS)

    def test_fee_question(self):
        self.assertEqual(self.clf.classify("how do I pay fees?"), "fees")

    def test_library_hours(self):
        self.assertEqual(self.clf.classify("when does the library close"), "library_hours")

    def test_location(self):
        self.assertEqual(self.clf.classify("where is the canteen"), "location")

    def test_advisor_question(self):
        self.assertEqual(self.clf.classify("who is my faculty advisor"), "advisor")

    def test_topics_command(self):
        self.assertIn("Library Hours", respond(self.clf, "topics"))
        self.assertIn("Fees", respond(self.clf, "help"))

    def test_unknown_question_falls_back(self):
        self.assertIsNone(self.clf.classify("what's the weather"))

    def test_math_query_falls_back(self):
        self.assertIsNone(self.clf.classify("what's 2+2"))

    def test_ask_gemini_without_key_returns_none(self):
        self.assertIsNone(ask_gemini("what is the weather", api_key=""))

    def test_digits_are_kept(self):
        self.assertEqual(tokenize("Block 3"), ["block", "3"])

    def test_plural_normalized(self):
        self.assertEqual(tokenize("exams"), ["exam"])


if __name__ == "__main__":
    # If run via 'streamlit run assistant.py', launch the web UI; otherwise launch CLI
    try:
        import streamlit as st
        is_streamlit = st.runtime.exists()
    except Exception:
        is_streamlit = False

    if is_streamlit:
        run_streamlit()
    else:
        main()
