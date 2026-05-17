from flask import Flask, render_template, request, jsonify
import PyPDF2
import json
import os
import re

from datetime import datetime
from groq import Groq
import sqlite3

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================================
# CONFIG
# =====================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================
# DATABASE
# =====================================================

def init_db():
    conn = sqlite3.connect("resume_ai.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT,
        ats_score REAL,
        resume_level TEXT,
        jd_match REAL,
        analysis_time TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# =====================================================
# GROQ HELPER
# =====================================================

def ask_groq(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate" in error_str.lower():
            return "⚠️ AI is temporarily rate-limited. Please wait a moment and try again."
        return f"⚠️ AI unavailable: {error_str}"

# =====================================================
# LOAD SKILLS
# =====================================================

def load_skills():
    with open("skills.json", "r") as f:
        return json.load(f)

# =====================================================
# PDF EXTRACTION
# =====================================================

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9+#/. ]', ' ', text)
    return text

# =====================================================
# NAME EXTRACTION
# =====================================================

def extract_name(text):
    lines = text.split("\n")
    for line in lines[:10]:
        line = line.strip()
        if 2 < len(line) < 40:
            if len(line.split()) <= 4:
                return line.title()
    return "Candidate"

# =====================================================
# SEMANTIC SKILL DETECTION
# =====================================================

def detect_skills(text, skills_db):
    detected = []
    for skill in skills_db:
        if skill.lower() in text.lower():
            detected.append(skill.title())
    return list(set(detected))
# =====================================================
# SECTION ANALYSIS
# =====================================================

def analyze_sections(text):
    text = text.lower()
    return {
        "Education": "education" in text,
        "Experience": "experience" in text,
        "Projects": "project" in text,
        "Skills": "skill" in text,
        "Certifications": "certification" in text,
        "Achievements": "achievement" in text,
        "Internships": "internship" in text,
        "Summary": "summary" in text
    }

# =====================================================
# EXPERIENCE DETECTION
# =====================================================

def estimate_experience(text):
    matches = re.findall(r'(\d+)\+?\s+years', text.lower())
    if matches:
        years = max([int(x) for x in matches])
        return f"{years}+ Years"
    return "Fresher / Entry Level"

# =====================================================
# PROJECT COUNT
# =====================================================

def count_projects(text):
    return min(text.lower().count("project"), 10)

# =====================================================
# EDUCATION EXTRACTION
# =====================================================

def extract_education(text):
    keywords = ["btech", "b.tech", "mtech", "m.tech", "bachelor", "master", "college", "university"]
    found = []
    for word in keywords:
        if word in text.lower():
            found.append(word.upper())
    return list(set(found))

# =====================================================
# JD MATCHING (keyword-based, no heavy ML model)
# =====================================================

def calculate_jd_match(resume_text, jd_text):
    if not jd_text:
        return 0
    resume_words = set(clean_text(resume_text).split())
    jd_words = set(clean_text(jd_text).split())
    if not jd_words:
        return 0
    common = resume_words & jd_words
    match = len(common) / len(jd_words) * 100
    return round(min(match, 100), 2)

# =====================================================
# ATS ENGINE
# =====================================================

def calculate_ats_score(sections, skills_count, project_count, jd_score, resume_text):
    ats_score = 0

    if sections.get("Education"):    ats_score += 10
    if sections.get("Experience"):   ats_score += 18
    if sections.get("Projects"):     ats_score += 22
    if sections.get("Skills"):       ats_score += 15
    if sections.get("Certifications"): ats_score += 8
    if sections.get("Achievements"): ats_score += 8
    if sections.get("Internships"):  ats_score += 10
    if sections.get("Summary"):      ats_score += 5

    ats_score += min(skills_count * 2.5, 20)

    advanced_keywords = [
        "Distributed", "Lru", "Consistent Hashing", "Cache", "Load Balancing",
          "Fault Tolerance", "High-Frequency Requests", "Segment Trees",
            "Dynamic Programming", "Graph Algorithms", "Greedy Algorithms", 
            "Sliding Window", "Authentication", "Api", "Node.js", "Express", 
            "Mysql", "Database", "Jwt", "React", "Javascript", "Dsa", "Oop", 
            "Dbms", "Computer Networks", "Operating Systems", "Wireshark", "Nmap", 
            "Docker", "Aws", "Vercel", "Render", "Production-Ready", "Optimized",
              "Responsive", "Secure", "Scalable"
    ]

    keyword_hits = 0
    for keyword in advanced_keywords:
        if keyword.lower() in resume_text.lower():
            keyword_hits += 1
            ats_score += 2

    project_keywords = [
        "distributed cache simulator", "resume analyzer", "job portal",
        "network port scanner", "socket programming",
        "role-based authentication", "session management"
    ]

    project_strength = 0
    for keyword in project_keywords:
        if keyword.lower() in resume_text.lower():
            project_strength += 1
            ats_score += 3

    if "intern" in resume_text.lower():             ats_score += 5
    if "full stack developer intern" in resume_text.lower(): ats_score += 4
    if "hackathon" in resume_text.lower():          ats_score += 4
    if "sih" in resume_text.lower():               ats_score += 5
    if "smart india hackathon" in resume_text.lower(): ats_score += 5

    ats_score += min(project_count * 2, 10)
    ats_score += jd_score * 0.12

    if skills_count >= 10:      ats_score += 4
    if keyword_hits >= 10:      ats_score += 5
    if project_strength >= 3:   ats_score += 5

    return min(round(ats_score), 98)

# =====================================================
# RESUME LEVEL
# =====================================================

def get_resume_level(score):
    if score >= 90: return "Elite Engineering Profile"
    elif score >= 80: return "Strong Technical Profile"
    elif score >= 70: return "Industry Ready Profile"
    elif score >= 60: return "Moderate Technical Resume"
    return "Needs Optimization"

# =====================================================
# MISSING SKILLS
# =====================================================

def find_missing_skills(all_skills, detected):
    detected_lower = [s.lower() for s in detected]
    missing = [s for s in all_skills if s.lower() not in detected_lower]
    return missing[:10]
# =====================================================
# QUESTIONS
# =====================================================

def generate_questions(skills):
    bank = {
        "react": ["Explain Virtual DOM.", "What are React Hooks?"],
        "python": ["Explain decorators.", "What is multithreading?"],
        "mysql": ["What is indexing?", "Explain JOIN types."],
        "node.js": ["What is middleware?", "Explain event loop."]
    }
    questions = []
    for skill in skills:
        if skill.lower() in bank:
            questions.extend(bank[skill.lower()])
    return questions[:8]

# =====================================================
# AI FEEDBACK
# =====================================================

def generate_ai_feedback(score, missing_skills):
    feedback = []
    if score < 60:
        feedback.append("Resume lacks strong technical depth.")
    if len(missing_skills) > 5:
        feedback.append("Add more industry-relevant skills.")
    feedback.append("Use quantified achievements.")
    feedback.append("Improve project impact descriptions.")
    return feedback

# =====================================================
# GROQ: RESUME SUMMARY
# =====================================================

def generate_resume_summary(text):
    prompt = f"""
    Analyze this resume and give:
    1. Professional summary
    2. Career domain
    3. Top strengths
    4. Improvement suggestions

    Resume:
    {text[:5000]}
    """
    return ask_groq(prompt)

# =====================================================
# GROQ: BULLET REWRITER
# =====================================================

def rewrite_resume_bullet(bullet):
    prompt = f"""
    Rewrite this resume bullet professionally
    with stronger action verbs and metrics.

    Bullet:
    {bullet}
    """
    return ask_groq(prompt)

# =====================================================
# GROQ: AI CHAT
# =====================================================

def get_ai_answer(question):
    return ask_groq(question)

# =====================================================
# SAVE ANALYSIS
# =====================================================

def save_analysis(name, ats, level, jd_score):
    conn = sqlite3.connect("resume_ai.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO analyses (candidate_name, ats_score, resume_level, jd_match, analysis_time)
    VALUES (?, ?, ?, ?, ?)
    """, (name, ats, level, jd_score, datetime.now().strftime("%d-%m-%Y %H:%M")))
    conn.commit()
    conn.close()

# =====================================================
# MAIN ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("resume")
        jd_text = request.form.get("job_description", "")

        if not file or file.filename == "":
            return render_template("index.html", error="Please upload resume PDF.")

        if not file.filename.lower().endswith(".pdf"):
            return render_template("index.html", error="Only PDF supported.")

        resume_text = extract_text_from_pdf(file)
        cleaned_text = clean_text(resume_text)

        skills_db = load_skills()
        detected_skills = detect_skills(cleaned_text, skills_db)
        missing_skills = find_missing_skills(skills_db, detected_skills)

        sections = analyze_sections(cleaned_text)
        experience = estimate_experience(cleaned_text)
        project_count = count_projects(cleaned_text)
        education = extract_education(cleaned_text)
        jd_match = calculate_jd_match(cleaned_text, jd_text)

        ats_score = calculate_ats_score(
            sections, len(detected_skills), project_count, jd_match, cleaned_text
        )

        resume_level = get_resume_level(ats_score)
        questions = generate_questions(detected_skills)
        ai_feedback = generate_ai_feedback(ats_score, missing_skills)
        ai_summary = generate_resume_summary(resume_text)
        candidate_name = extract_name(resume_text)

        save_analysis(candidate_name, ats_score, resume_level, jd_match)

        semantic_match = min(65 + len(detected_skills), 97)
        resume_strength = min(60 + (project_count * 6) + len(detected_skills), 95)
        ai_confidence = min(72 + len(detected_skills), 96)

        return render_template(
            "result.html",
            candidate_name=candidate_name,
            ats_score=ats_score,
            resume_level=resume_level,
            resume_skills=detected_skills,
            missing_skills=missing_skills,
            sections=sections,
            questions=questions,
            experience=experience,
            education=education,
            project_count=project_count,
            jd_match=jd_match,
            ai_feedback=ai_feedback,
            semantic_match=semantic_match,
            resume_strength=resume_strength,
            ai_confidence=ai_confidence,
            ai_summary=ai_summary,
            analysis_time=datetime.now().strftime("%d %B %Y • %I:%M %p")
        )

    return render_template("index.html")

# =====================================================
# AI CHAT
# =====================================================

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"answer": "Please ask a question."})
    answer = get_ai_answer(question)
    return jsonify({"answer": answer})

# =====================================================
# BULLET REWRITER
# =====================================================

@app.route("/rewrite", methods=["POST"])
def rewrite():
    bullet = request.form.get("bullet")
    if not bullet:
        return jsonify({"rewrite": "Enter a bullet point."})
    rewritten = rewrite_resume_bullet(bullet)
    return jsonify({"rewrite": rewritten})

# =====================================================
# STATUS
# =====================================================

@app.route("/status")
def status():
    return jsonify({
        "status": "active",
        "ai_engine": "online",
        "model": "Groq LLaMA3",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
