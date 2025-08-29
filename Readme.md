# ROG Xbox Ally Chatbot

A **Retrieval-Augmented Generation (RAG)** based chatbot that answers questions about the **Xbox ROG Ally handheld device** using pre-stored data in PostgreSQL.  
It leverages **semantic embeddings** from `SentenceTransformers` and a **Groq LLM** to generate accurate, context-aware answers.

---

## 🛠 Features

- **Semantic Search:** Retrieves the most relevant data chunks for user queries using vector similarity.
- **RAG (Retrieval-Augmented Generation):** Combines retrieved data with LLM to generate natural, context-aware answers.
- **Interactive Web Frontend:** Sleek chat UI with gradient backgrounds, user/bot message distinction, and responsive design.
- **Sources Display:** Shows top relevant sources with similarity scores.
- **PostgreSQL Backend:** Stores Xbox ROG Ally data with embeddings for semantic querying.
- **Dockerized PostgreSQL (optional):** Easy setup with PostgreSQL container.

---

## 🗂 Directory Structure

devmeh19-chatbotcx/
├── README.md
├── main.py # FastAPI application
├── requirements.txt # Python dependencies
├── start.bat # Script to run the app locally
├── scraper.py # Optional web scraping utilities
├── advanced_scraper.py # Advanced scraping scripts for Xbox data
├── run_enhanced_chatbot.bat
├── run_scraper.bat
└── scraper_requirements.txt

yaml
Copy code

---

## ⚡ Technologies Used

- **Backend & API:** Python, FastAPI, Uvicorn  
- **Database:** PostgreSQL (or Dockerized PostgreSQL container)  
- **Embeddings & Semantic Search:** `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **LLM Integration:** Groq API  
- **Frontend:** HTML, CSS, JavaScript (interactive chat UI)  
- **Deployment:** Configured for cloud platforms like Render  

---

## Images

![Description](images/1.png)
![Description](images/2.png)
![Description](images/3.png)
![Description](images/4.png)


## 🔧 Setup Instructions

###1. Clone the Repository

```bash
git clone https://github.com/devmeh19/devmeh19-chatbotcx.git
cd devmeh19-chatbotcx
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. PostgreSQL Setup
You can run a local PostgreSQL instance or use Docker.

Using Docker:
bash
Copy code
docker run --name xbox-chatbot-db \
    -e POSTGRES_PASSWORD=YourPassword \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_DB=chatbotdata \
    -p 5432:5432 -d postgres
Database Table (items_xbox):
sql
Copy code
CREATE TABLE items_xbox (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384)  -- Matches SentenceTransformer embedding dimension
);
4. Generate and Insert Embeddings
Use SentenceTransformer to encode Xbox ROG Ally data into vector embeddings and store them in PostgreSQL.

python
Copy code
from sentence_transformers import SentenceTransformer
import psycopg

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to PostgreSQL
conn = psycopg.connect("postgresql://postgres:YourPassword@localhost:5432/chatbotdata")
cursor = conn.cursor()

# Sample Xbox data
texts = [
    "ROG Xbox Ally has a 7-inch display...",
    "The device runs on Windows 11..."
]

# Generate embeddings and insert into DB
for text in texts:
    embedding = model.encode([text])[0].tolist()
    cursor.execute("INSERT INTO items_xbox (text, embedding) VALUES (%s, %s)", (text, embedding))

conn.commit()
cursor.close()
conn.close()
5. Environment Variables
Create a .env file in the root directory:

env
Copy code
DATABASE_URL1=postgresql://postgres:YourPassword@localhost:5432/chatbotdata
GROQ_API_KEY1=your_groq_api_key
GROQ_MODEL1=llama-3.3-70b-versatile
PORT=8080
6. Run the Application
bash
Copy code
python main.py
Open your browser at http://localhost:8080 to interact with the chatbot.




  🧠 How It Works
User Query: User submits a question through the web frontend.

Semantic Retrieval: Query is encoded using SentenceTransformer, then the system retrieves top k similar chunks from PostgreSQL using vector similarity.

Contextual Answering: Retrieved chunks are fed to the Groq LLM, which generates a detailed, context-aware response.

Response & Sources: The chatbot returns the answer along with relevant sources and similarity scores.

This ensures answers are grounded in real data, improving reliability and transparency.

🎨 Frontend Highlights
Modern, blurred chat container with gradient backgrounds.

User messages are color-coded differently from bot messages.

Sources are displayed with similarity scores for credibility.

Fully responsive design for desktop and mobile screens.

🚀 Next Steps / Improvements
Integrate live scraping of Xbox website for up-to-date specs.

Add multilingual support for global users.

Deploy to Render/Heroku with CI/CD pipelines.

Expand the database with FAQs, troubleshooting guides, and reviews for richer answers.