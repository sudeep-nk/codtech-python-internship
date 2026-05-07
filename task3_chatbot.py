# ============================================================
# CODTECH INTERNSHIP - TASK 3
# AI Chatbot with NLP
# Author: [Your Name]
# Description: A college information chatbot using NLTK,
#              TF-IDF for intent matching, and SpaCy for NER.
#              Includes conversation memory and Streamlit UI.
# ============================================================

import nltk
import numpy as np
import random
import string
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────
KNOWLEDGE_BASE = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy", "what's up"],
        "responses": [
            "Hello! 👋 Welcome to College Assistant. How can I help you today?",
            "Hi there! I'm your college information assistant. What would you like to know?",
            "Hey! Great to see you. Ask me anything about admissions, courses, or campus life!",
        ]
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "take care", "exit", "quit", "later"],
        "responses": [
            "Goodbye! Best of luck with your studies! 🎓",
            "See you later! Feel free to come back anytime.",
            "Take care! All the best! 👋",
        ]
    },
    "admission": {
        "patterns": [
            "how to apply", "admission process", "eligibility", "entrance exam",
            "how do i get admission", "admission criteria", "apply for admission",
            "when is last date", "application form", "admission requirements"
        ],
        "responses": [
            "📋 **Admission Process:**\n"
            "1. Fill the online application at college website\n"
            "2. Upload required documents (10th, 12th marksheets, ID proof)\n"
            "3. Pay the application fee of ₹500\n"
            "4. Appear for entrance test (if applicable)\n"
            "5. Merit list will be published within 2 weeks\n"
            "Last date for applications: **July 31st**. Need more details?",
        ]
    },
    "courses": {
        "patterns": [
            "what courses", "available programs", "which courses", "list of courses",
            "departments", "branches", "what can i study", "programs offered"
        ],
        "responses": [
            "🎓 **Courses Offered:**\n"
            "**UG Programs:** B.Tech (CS, EC, ME, Civil), BCA, B.Sc\n"
            "**PG Programs:** M.Tech, MCA, MBA, M.Sc\n"
            "**Research:** PhD in CS, Electronics, Management\n"
            "All programs are AICTE/UGC approved. Which course interests you?",
        ]
    },
    "fees": {
        "patterns": [
            "fee structure", "how much fees", "tuition fee", "cost", "fee details",
            "how much does it cost", "scholarship", "fee waiver"
        ],
        "responses": [
            "💰 **Fee Structure (Per Year):**\n"
            "• B.Tech: ₹80,000 – ₹1,20,000\n"
            "• MCA: ₹60,000 – ₹90,000\n"
            "• MBA: ₹70,000 – ₹1,00,000\n"
            "• BCA: ₹40,000 – ₹60,000\n\n"
            "🎁 **Scholarships available** for merit students (up to 50% fee waiver).\n"
            "Contact the accounts office for exact details.",
        ]
    },
    "placement": {
        "patterns": [
            "placement", "jobs", "recruiting companies", "campus placement",
            "salary package", "placements record", "companies visit", "hiring"
        ],
        "responses": [
            "🏢 **Placement Highlights:**\n"
            "• 95%+ placement rate for CS and IT students\n"
            "• Average package: ₹5.5 LPA | Highest: ₹28 LPA\n"
            "• Top recruiters: TCS, Infosys, Wipro, Amazon, Google, Cognizant\n"
            "• Dedicated Training & Placement Cell\n"
            "• Pre-placement training: Aptitude, Coding, Communication",
        ]
    },
    "hostel": {
        "patterns": [
            "hostel", "accommodation", "room", "dormitory", "stay on campus",
            "hostel fees", "hostel facilities", "food in hostel"
        ],
        "responses": [
            "🏠 **Hostel Facilities:**\n"
            "• Separate boys' and girls' hostels\n"
            "• 24/7 Wi-Fi, laundry, mess facility\n"
            "• Fee: ₹50,000/year (includes food)\n"
            "• Gym, TV room, indoor games available\n"
            "• 24-hour security & CCTV surveillance",
        ]
    },
    "faculty": {
        "patterns": [
            "faculty", "professors", "teachers", "staff", "who teaches",
            "faculty details", "teaching staff", "best professors"
        ],
        "responses": [
            "👨‍🏫 **Faculty:**\n"
            "• 200+ highly qualified faculty members\n"
            "• 70% have PhD degrees\n"
            "• Average experience: 12 years\n"
            "• Faculty from IITs, NITs, and top universities\n"
            "• Active research publications in international journals",
        ]
    },
    "library": {
        "patterns": [
            "library", "books", "reading", "e-library", "digital library",
            "library timings", "study material"
        ],
        "responses": [
            "📚 **Library:**\n"
            "• Collection of 50,000+ books\n"
            "• Subscribed to IEEE, Springer, Elsevier journals\n"
            "• E-library with 24/7 online access\n"
            "• Timings: 8 AM – 9 PM (Mon–Sat)\n"
            "• Separate reading halls for quiet study",
        ]
    },
    "exam": {
        "patterns": [
            "exam", "examination", "schedule", "time table", "syllabus",
            "results", "marks", "grade", "cgpa", "backlog"
        ],
        "responses": [
            "📝 **Examinations:**\n"
            "• 2 semesters per year (June–Nov, Dec–May)\n"
            "• Internal: 40 marks | External: 60 marks\n"
            "• Results declared within 30 days after exam\n"
            "• Minimum passing: 40% in each subject\n"
            "• Re-exam (supplementary) allowed for backlogs",
        ]
    },
    "sports": {
        "patterns": [
            "sports", "games", "cricket", "football", "basketball",
            "sports facilities", "playground", "gym", "fitness"
        ],
        "responses": [
            "⚽ **Sports & Fitness:**\n"
            "• Cricket ground, football field, basketball courts\n"
            "• Indoor: Badminton, TT, Chess, Carrom\n"
            "• Gymnasium with modern equipment\n"
            "• Annual sports fest: SPORTOMANIA\n"
            "• Coaching available for national-level events",
        ]
    },
    "contact": {
        "patterns": [
            "contact", "phone number", "address", "email", "location",
            "how to reach", "where is college", "website"
        ],
        "responses": [
            "📞 **Contact Information:**\n"
            "• 📍 Address: College Road, Davangere, Karnataka – 577002\n"
            "• 📞 Phone: +91 98765 43210\n"
            "• 📧 Email: info@college.edu.in\n"
            "• 🌐 Website: www.college.edu.in\n"
            "• Office Hours: Mon–Sat, 9 AM – 5 PM",
        ]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thank u", "helpful", "great", "awesome", "nice"],
        "responses": [
            "You're welcome! 😊 Is there anything else I can help you with?",
            "Happy to help! Let me know if you have more questions.",
            "Glad I could assist! 🎓",
        ]
    }
}

# ─────────────────────────────────────────
# NLP PROCESSOR
# ─────────────────────────────────────────
class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer()
        self._build_index()

    def preprocess(self, text):
        """Tokenize, lowercase, remove stopwords, and lemmatize."""
        tokens = nltk.word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens
                  if t not in string.punctuation and t not in self.stop_words]
        return " ".join(tokens)

    def _build_index(self):
        """Build TF-IDF matrix from all patterns."""
        self.intents = []
        self.all_patterns = []
        for intent, data in KNOWLEDGE_BASE.items():
            for pattern in data["patterns"]:
                self.intents.append(intent)
                self.all_patterns.append(self.preprocess(pattern))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_patterns)

    def get_intent(self, user_input):
        """Find best matching intent using cosine similarity."""
        processed = self.preprocess(user_input)
        user_vec = self.vectorizer.transform([processed])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        if best_score < 0.15:
            return None, best_score
        return self.intents[best_idx], best_score


# ─────────────────────────────────────────
# CHATBOT ENGINE
# ─────────────────────────────────────────
class CollegeChatbot:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.history = []      # Conversation memory
        self.context = None    # Track last intent for follow-up

    def respond(self, user_input):
        """Generate a response for the given user input."""
        user_input = user_input.strip()
        if not user_input:
            return "Please type a question. I'm here to help! 😊"

        # Store in conversation history
        self.history.append({"role": "user", "text": user_input})

        # Detect intent
        intent, score = self.nlp.get_intent(user_input)

        if intent is None:
            response = (
                "🤔 I'm not sure I understand that. Could you rephrase?\n"
                "You can ask me about: **admissions, courses, fees, placements, "
                "hostel, faculty, exams, library, sports,** or **contact info**."
            )
        else:
            self.context = intent
            response = random.choice(KNOWLEDGE_BASE[intent]["responses"])

        self.history.append({"role": "bot", "text": response})
        return response

    def get_history(self):
        return self.history


# ─────────────────────────────────────────
# TERMINAL INTERFACE
# ─────────────────────────────────────────
def run_terminal():
    """Run chatbot in terminal mode."""
    bot = CollegeChatbot()
    print("\n" + "="*60)
    print("  🎓 COLLEGE ASSISTANT CHATBOT  (CodTech Task 3)")
    print("="*60)
    print("  Type your question below. Type 'bye' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        response = bot.respond(user_input)
        print(f"\nBot: {response}\n")
        if "Goodbye" in response or "See you later" in response or "Take care" in response:
            break


# ─────────────────────────────────────────
# STREAMLIT UI (run with: streamlit run task3_chatbot.py)
# ─────────────────────────────────────────
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="College Chatbot", page_icon="🎓", layout="centered")
    st.title("🎓 College Assistant Chatbot")
    st.caption("Powered by NLP (NLTK + TF-IDF) | CodTech Internship Task 3")

    # Initialize session state
    if "bot" not in st.session_state:
        st.session_state.bot = CollegeChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! 👋 I'm your College Assistant. Ask me about admissions, courses, fees, placements, and more!"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask me anything about the college..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = st.session_state.bot.respond(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Sidebar with quick topics
    with st.sidebar:
        st.header("💡 Quick Topics")
        topics = ["Admission Process", "Fee Structure", "Placements", "Hostel",
                  "Courses Offered", "Exam Schedule", "Library", "Contact Info"]
        for topic in topics:
            if st.button(topic, use_container_width=True):
                response = st.session_state.bot.respond(topic)
                st.session_state.messages.append({"role": "user", "content": topic})
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()
    else:
        # Try Streamlit; fallback to terminal
        try:
            import streamlit as st
            run_streamlit()
        except Exception:
            run_terminal()
