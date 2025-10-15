import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Courses": "BSIT",
    "os_perc": 78,
    "algo_perc": 85,
    "prog_perc": 91,
    "se_perc": 80,
    "cn_perc": 76,
    "es_perc": 88,
    "ca_perc": 83,
    "math_perc": 89,
    "comm_perc": 72,
    "hours_working": 7,
    "hackathons": 5,
    "logical_quotient": 3,
    "coding_skills": 4,
    "public_speaking": 2,
    "memory_score": 8,
    "interested_subjects": "Software Engineering",
    "career_area": "cloud computing",
    "company_type": "Product based",
    "books": "Technical",
    "behavior": "gentle",
    "management_tech": "Technical",
    "salary_work": "work",
    "team_exp": "yes",
    "work_style": "smart worker",
    "relationship": "no",
    "introvert": "yes",
    "seniors_input": "yes",
    "gaming_interest": "no"
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
