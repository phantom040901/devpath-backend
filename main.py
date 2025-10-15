from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from collections import defaultdict

# ---------------- Load models ----------------
preprocessor = joblib.load("preprocessor.pkl")
job_profiles = joblib.load("job_profiles.pkl")

app = FastAPI(title="Career Path Predictor API")

# ---------------- Enable CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Input Schema ----------------
class StudentProfile(BaseModel):
    courses: str
    os_perc: int
    algo_perc: int
    prog_perc: int
    se_perc: int
    cn_perc: int
    es_perc: int
    ca_perc: int
    math_perc: int
    comm_perc: int
    hours_working: int
    hackathons: int
    logical_quotient: int
    coding_skills: int
    public_speaking: int
    memory_score: int
    interested_subjects: str
    career_area: str
    company_type: str
    books: str
    behavior: str
    management_tech: str
    salary_work: str
    team_exp: str
    work_style: str
    relationship: str
    introvert: str
    seniors_input: str
    gaming_interest: str

# ---------------- Category Mapping ----------------
ROLE_TO_CATEGORY = {
    # Software Development
    "Software Engineer": "Software Development",
    "Software Developer": "Software Development",
    "Applications Developer": "Software Development",
    "Mobile Applications Developer": "Software Development",
    "Web Developer": "Software Development",
    "Programmer Analyst": "Software Development",
    "UX Designer": "Software Development",
    "Design & UX": "Software Development",

    # Data & Analytics
    "Database Administrator": "Data & Analytics",
    "Database Developer": "Data & Analytics",
    "Database Manager": "Data & Analytics",
    "Data Architect": "Data & Analytics",
    "Business Intelligence Analyst": "Data & Analytics",
    "E-Commerce Analyst": "Data & Analytics",

    # Networking & Security
    "Network Security Engineer": "Networking & Security",
    "Network Engineer": "Networking & Security",
    "Network Security Administrator": "Networking & Security",
    "Systems Security Administrator": "Networking & Security",
    "Information Security Analyst": "Networking & Security",

    # Quality Assurance & Testing
    "Quality Assurance Associate": "Quality Assurance & Testing",
    "Software Quality Assurance (QA) / Testing": "Quality Assurance & Testing",

    # IT Management
    "Project Manager": "IT Management",
    "Information Technology Manager": "IT Management",
    "CRM Business Analyst": "IT Management",
    "Business Systems Analyst": "IT Management",
    "Solutions Architect": "IT Management",

    # Technical Support
    "Technical Support": "Technical Support",
    "Technical Services/Help Desk/Tech Support": "Technical Support",

    # Other / Specialized
    "Information Technology Auditor": "Specialized",
    "Portal Administrator": "Specialized",
    "CRM Technical Developer": "Specialized",
}

# ---------------- Enhanced Feature Weights ----------------
FEATURE_WEIGHTS = {
    "Acedamic percentage in Operating Systems": 1.3,
    "percentage in Algorithms": 1.4,
    "Percentage in Programming Concepts": 1.5,
    "Percentage in Software Engineering": 1.3,
    "Percentage in Computer Networks": 1.5,  # Increased for security
    "Percentage in Electronics Subjects": 0.9,
    "Percentage in Computer Architecture": 1.0,
    "Percentage in Mathematics": 1.1,
    "Logical quotient rating": 1.8,  # Increased importance
    "coding skills rating": 1.6,
    "Percentage in Communication skills": 1.4,
    "public speaking points": 1.3,
    "Management or Technical": 1.5,
    "interested career area": 2.5,  # CRITICAL - Much higher weight
    "Type of company want to settle in?": 1.5,
    "Interested subjects": 2.3,  # CRITICAL - Much higher weight
    "hard/smart worker": 1.2,
    "worked in teams ever?": 1.3,
    "hackathons": 1.4,
    "Hours working per day": 0.9,
    "Interested Type of Books": 0.6,
    "Gentle or Tuff behaviour?": 0.7,
    "Salary/work": 0.8,
    "In a Realtionship?": 0.3,
    "Introvert": 0.5,
    "Taken inputs from seniors or elders": 0.5,
    "interested in games": 0.4,
    "memory capability score": 0.8
}

# ---------------- Helper Functions ----------------
def make_dataframe(profile: StudentProfile) -> pd.DataFrame:
    return pd.DataFrame([{
        'Courses': profile.courses,
        'Acedamic percentage in Operating Systems': profile.os_perc,
        'percentage in Algorithms': profile.algo_perc,
        'Percentage in Programming Concepts': profile.prog_perc,
        'Percentage in Software Engineering': profile.se_perc,
        'Percentage in Computer Networks': profile.cn_perc,
        'Percentage in Electronics Subjects': profile.es_perc,
        'Percentage in Computer Architecture': profile.ca_perc,
        'Percentage in Mathematics': profile.math_perc,
        'Percentage in Communication skills': profile.comm_perc,
        'Hours working per day': profile.hours_working,
        'Logical quotient rating': profile.logical_quotient,
        'hackathons': profile.hackathons,
        'coding skills rating': profile.coding_skills,
        'public speaking points': profile.public_speaking,
        'memory capability score': profile.memory_score,
        'Interested subjects': profile.interested_subjects,
        'interested career area': profile.career_area,
        'Type of company want to settle in?': profile.company_type,
        'Taken inputs from seniors or elders': profile.seniors_input,
        'interested in games': profile.gaming_interest,
        'Interested Type of Books': profile.books,
        'In a Realtionship?': profile.relationship,
        'Gentle or Tuff behaviour?': profile.behavior,
        'Management or Technical': profile.management_tech,
        'Salary/work': profile.salary_work,
        'hard/smart worker': profile.work_style,
        'worked in teams ever?': profile.team_exp,
        'Introvert': profile.introvert
    }])

def apply_career_area_bonus(scores: np.ndarray, job_roles: list, career_area: str, interested_subjects: str, profile: StudentProfile) -> np.ndarray:
    """Apply significant bonus to jobs matching career area, interests, and profile characteristics"""
    adjusted_scores = scores.copy()
    
    # Define keyword mappings
    security_keywords = ['security', 'network security', 'information security', 'systems security']
    data_keywords = ['data', 'database', 'business intelligence', 'analytics']
    dev_keywords = ['software', 'developer', 'applications', 'web', 'mobile', 'programmer']
    design_keywords = ['ux', 'design', 'ui']
    
    # Design profile indicators
    is_design_profile = (
        profile.comm_perc >= 90 and 
        profile.public_speaking >= 4 and
        profile.management_tech.lower() == 'management' and
        profile.company_type.lower() in ['product development', 'product based']
    )
    
    for i, job in enumerate(job_roles):
        job_lower = job.lower()
        bonus = 0
        
        # Design role matching
        if is_design_profile and any(kw in job_lower for kw in design_keywords):
            bonus += 0.35  # 35% bonus for design profiles
            
        # Career area matching
        if career_area.lower() == 'security':
            if any(kw in job_lower for kw in security_keywords):
                bonus += 0.25
        elif career_area.lower() in ['data engineering', 'database']:
            if any(kw in job_lower for kw in data_keywords):
                bonus += 0.25
        elif career_area.lower() in ['system developer', 'developer']:
            # Only boost dev roles if NOT a design profile
            if not is_design_profile and any(kw in job_lower for kw in dev_keywords):
                bonus += 0.20
        
        # Interested subjects matching
        if interested_subjects.lower() == 'hacking':
            if any(kw in job_lower for kw in security_keywords):
                bonus += 0.20
        elif interested_subjects.lower() in ['networks', 'networking']:
            if 'network' in job_lower:
                bonus += 0.20
        elif interested_subjects.lower() in ['software engineering', 'design']:
            if any(kw in job_lower for kw in design_keywords):
                bonus += 0.15
        
        adjusted_scores[i] = min(1.0, adjusted_scores[i] * (1 + bonus))
    
    return adjusted_scores

def weighted_cosine_similarity_vectorized(user_vec: np.ndarray, job_matrix: np.ndarray, feature_names) -> np.ndarray:
    weights = np.array([FEATURE_WEIGHTS.get(f, 1.0) for f in feature_names], dtype=float)
    w_user = user_vec * weights
    w_jobs = job_matrix * weights
    sims = cosine_similarity([w_user], w_jobs)[0]
    return sims

def adjust_scores_with_gaps(raw_scores, min_gap=3):
    """Create meaningful gaps between scores, rounded to whole numbers"""
    scores = [round(float(s) * 100) for s in raw_scores]  # <-- round to whole %
    
    # Sort and create gaps
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    adjusted = scores.copy()
    
    for rank in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[rank - 1]
        curr_idx = sorted_indices[rank]
        
        # Ensure minimum gap
        if adjusted[curr_idx] >= adjusted[prev_idx] - min_gap:
            adjusted[curr_idx] = max(0, adjusted[prev_idx] - min_gap)
    
    # Cap maximum at 95%
    max_score = max(adjusted)
    if max_score > 95:
        scale_factor = 95 / max_score
        adjusted = [round(s * scale_factor) for s in adjusted]
    
    return adjusted

def run_prediction(profile: StudentProfile) -> Dict:
    # Build input
    input_df = make_dataframe(profile)
    processed_input = preprocessor.transform(input_df)

    feature_names = preprocessor.get_feature_names_out()
    user_vec = processed_input[0] if processed_input.ndim > 1 else processed_input
    job_matrix = job_profiles.values
    job_roles = list(job_profiles.index)

    # 1) Raw weighted cosine similarity
    raw_scores = weighted_cosine_similarity_vectorized(user_vec, job_matrix, feature_names)

    # 2) Apply career area and interest bonuses
    boosted_scores = apply_career_area_bonus(
        raw_scores, 
        job_roles, 
        profile.career_area, 
        profile.interested_subjects,
        profile
    )

    # 3) Adjust scores with gaps (now integers)
    adjusted_scores = adjust_scores_with_gaps(boosted_scores, min_gap=3)

    # 4) Package results with whole numbers
    job_records = []
    for i, jr in enumerate(job_roles):
        job_records.append({
            "job_role": jr,
            "category": ROLE_TO_CATEGORY.get(jr, "Other"),
            "match_score": f"{adjusted_scores[i]}%"  # no decimals
        })

    # Sort and take top 3
    job_records = sorted(job_records, key=lambda x: int(x["match_score"].strip('%')), reverse=True)[:3]

    return {"job_matches": job_records}


# ---------------- Endpoints ----------------
@app.post("/predict")
def predict(profile: StudentProfile):
    return {"recommendations": run_prediction(profile)}

@app.get("/predict/random")
def predict_random():
    profile = StudentProfile(
        os_perc=random.randint(75, 100),
        algo_perc=random.randint(75, 100),
        prog_perc=random.randint(75, 100),
        se_perc=random.randint(75, 100),
        cn_perc=random.randint(75, 100),
        es_perc=random.randint(75, 95),
        ca_perc=random.randint(75, 95),
        math_perc=random.randint(75, 95),
        comm_perc=random.randint(75, 100),
        hours_working=random.randint(4, 12),
        hackathons=random.randint(0, 10),
        logical_quotient=random.randint(1, 5),
        coding_skills=random.randint(1, 5),
        public_speaking=random.randint(1, 5),
        memory_score=random.randint(5, 10),
        courses=random.choice(['BSIT']),
        interested_subjects=random.choice(['networks', 'cloud computing', 'hacking',
                                           'parallel computing', 'Software Engineering',
                                           'Computer Architecture', 'IOT', 'Management']),
        career_area=random.choice(['cloud computing', 'data engineering', 'system developer',
                                   'security', 'Business process analyst', 'testing']),
        company_type=random.choice(['product development', 'Product based', 'Service Based',
                                    'Cloud Services', 'Web Services', 'SAaaS services']),
        books=random.choice(['Cookbooks', 'Technical', 'Science fiction', 'Mystery',
                             'Self help', 'Fantasy', 'Biographies']),
        behavior=random.choice(['stubborn', 'gentle']),
        management_tech=random.choice(['Technical', 'Management']),
        salary_work=random.choice(['salary', 'work']),
        team_exp=random.choice(['yes', 'no']),
        work_style=random.choice(['hard worker', 'smart worker']),
        relationship=random.choice(['yes', 'no']),
        introvert=random.choice(['yes', 'no']),
        seniors_input=random.choice(['yes', 'no']),
        gaming_interest=random.choice(['yes', 'no']),
    )
    return {
        "input": profile.dict(),
        "recommendations": run_prediction(profile)
    }