# Campus Virtual Assistant

A simple campus FAQ virtual assistant built for AIP203 (Artificial Intelligence Lab). It uses a Multinomial Naive Bayes classifier from scratch to answer questions about campus topics, with optional Google Gemini API support for general/open-ended questions.

## Topics Covered
- Library hours and weekend timings
- Tuition and hostel fee payment deadlines
- Midterm and end-semester exam schedules
- Campus locations (canteen, admin office, blocks)
- Faculty advisors and mentor details

---

## How to Run

### 1. Run the Web Interface (Streamlit)
```bash
streamlit run assistant.py
```
Opens the chat interface in your browser (default: `http://localhost:8501`).

### 2. Run in the Terminal (CLI)
```bash
python assistant.py
```
- Type your questions directly in the terminal.
- Type `topics` or `help` to see sample questions.
- Type `quit` or `exit` to close the assistant.

### 3. Run Unit Tests
```bash
python -m unittest assistant.py
```

---

## (Optional) Gemini API Setup
For questions beyond the standard campus FAQs (e.g., math, general questions), create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```
Or paste your API key in the sidebar of the Streamlit web interface. If no key is set, the assistant runs using only the local Naive Bayes classifier.
