# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import psycopg2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# --- Database Connection ---
DATABASE_URL = "dbname='manabi_db' user='user' password='password' host='localhost' port='5432'"
try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Database connection established.")
except Exception as e:
    print("Database connection failed:", e)

# Create tables if they don't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        academic_background TEXT,
        career_goals TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS progress (
        progress_id SERIAL PRIMARY KEY,
        user_id INTEGER,
        progress_data TEXT
    )
""")
conn.commit()


# --- AI Models ---

# For generative AI, we use GPT-2 to generate a personalized summary.
text_gen_model = pipeline("text-generation", model="gpt2")

# For recommendation, we use a simple NearestNeighbors model.
# Create a dummy dataset of numeric vectors representing courses (for demonstration).
dummy_vectors = np.arange(0, 100).reshape(-1, 1)  # 100 dummy course vectors
recommendation_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
recommendation_model.fit(dummy_vectors)

# --- Pydantic Models ---
class UserProfile(BaseModel):
    user_id: int = None  # Optional on creation; assigned by DB
    academic_background: str
    certifications: List[str]
    career_goals: str
    learning_preferences: List[str]

# --- API Endpoints ---

@app.post("/profile/create")
def create_profile(profile: UserProfile):
    try:
        cursor.execute(
            "INSERT INTO users (academic_background, career_goals) VALUES (%s, %s) RETURNING user_id",
            (profile.academic_background, profile.career_goals)
        )
        conn.commit()
        user_id = cursor.fetchone()[0]
        return {"message": "User profile created successfully", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend_courses(profile: UserProfile):
    try:
        # For demonstration, convert career goals into a simple numeric value.
        # (In production, you’d use a proper text-embedding method.)
        vector_value = hash(profile.career_goals) % 100  
        vectorized_input = np.array([[vector_value]])
        # Find the nearest neighbors in our dummy dataset
        recommended_indices = recommendation_model.kneighbors(vectorized_input, return_distance=False)
        # Return as list (each index here represents a dummy course id)
        return {"recommendations": recommended_indices.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/progress/{user_id}")
def track_progress(user_id: int):
    try:
        cursor.execute("SELECT progress_data FROM progress WHERE user_id = %s", (user_id,))
        progress_data = cursor.fetchall()
        # Convert list of tuples to a simple list of progress strings.
        progress_list = [data[0] for data in progress_data]
        return {"progress": progress_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_summary")
def generate_summary(profile: UserProfile):
    try:
        # Create a prompt that summarizes the user's background and learning preferences.
        prompt = (
            f"User with academic background '{profile.academic_background}', "
            f"certifications {profile.certifications}, career goals '{profile.career_goals}', "
            f"and learning preferences {profile.learning_preferences} would benefit from a learning plan that includes"
        )
        generated = text_gen_model(prompt, max_length=100, num_return_sequences=1)
        summary = generated[0]['generated_text']
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
