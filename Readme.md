# ROG Xbox Ally Chatbot

A **Retrieval-Augmented Generation (RAG)** based chatbot that answers questions about the **Xbox ROG Ally handheld device** using pre-stored data in PostgreSQL. It leverages **semantic embeddings** from `SentenceTransformers` and a **Groq LLM** to generate precise, context-aware answers.  

---

## 🛠 Features

- **Semantic Search:** Uses vector embeddings to retrieve the most relevant chunks of data for a user query.
- **RAG (Retrieval-Augmented Generation):** Combines retrieved data with a language model to generate natural, accurate responses.
- **Interactive Web Frontend:** Built with **HTML, CSS, and JavaScript**, providing a sleek, gradient-based chat interface with user/bot distinction.
- **Sources Display:** Shows the top relevant sources for transparency and trust.
- **PostgreSQL Backend:** Stores Xbox ROG Ally data along with embeddings for semantic querying.
- **Dockerized PostgreSQL (optional):** Easy local setup using a PostgreSQL container.

---

## 🗂 Directory Structure

devmeh19-chatbotcx/
├── README.md
├── main.py # FastAPI app
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
- **Database:** PostgreSQL (pg container optional)  
- **Embeddings & Semantic Search:** `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **LLM Integration:** Groq API  
- **Frontend:** Vanilla HTML/CSS/JS with interactive chat UI  
- **Deployment:** Configured to run on cloud platforms like Render  

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/devmeh19/devmeh19-chatbotcx.git
cd devmeh19-chatbotcx
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. PostgreSQL Setup
You can either run a local PostgreSQL instance or use Docker:

Using Docker:
bash
Copy code
docker run --name xbox-chatbot-db -e POSTGRES_PASSWORD=YourPassword -e POSTGRES_USER=postgres -e POSTGRES_DB=chatbotdata -p 5432:5432 -d postgres
Database Table (items_xbox):
sql
Copy code
CREATE TABLE items_xbox (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384)  -- Matches SentenceTransformer embedding dimension
);
4. Generate and Insert Embeddings
Use SentenceTransformer to encode your Xbox ROG Ally data into vector embeddings:

python
Copy code
from sentence_transformers import SentenceTransformer
import psycopg

model = SentenceTransformer('all-MiniLM-L6-v2')
conn = psycopg.connect("postgresql://postgres:YourPassword@localhost:5432/chatbotdata")
cursor = conn.cursor()

texts = [
    "ROG Xbox Ally has a 7-inch display...",
    "The device runs on Windows 11..."
]

for text in texts:
    embedding = model.encode([text])[0].tolist()
    cursor.execute("INSERT INTO items_xbox (text, embedding) VALUES (%s, %s)", (text, embedding))

conn.commit()
cursor.close()
conn.close()
5. Environment Variables
Create a .env file with:

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
User Query: The user sends a question through the web frontend.

Semantic Retrieval: The query is encoded using SentenceTransformer. The system searches for the top k similar chunks in PostgreSQL using vector similarity.

Contextual Answering: The retrieved chunks are passed to Groq LLM to generate a detailed, context-aware answer.

Response & Sources: The chatbot returns the answer along with relevant sources and similarity scores.

This ensures that answers are grounded in real data, providing reliability and transparency.

🎨 Frontend Highlights
Modern, blurred chat container with gradient backgrounds.

User messages are color-coded differently from bot messages.

Sources are displayed with similarity scores to enhance credibility.

Responsive design for both desktop and mobile screens.

🚀 Next Steps / Improvements
Integrate live scraping of Xbox website for the latest specifications.

Add multilingual support for global users.

Deploy to Render/Heroku with proper CI/CD pipelines.

Expand database with FAQs, troubleshooting guides, and reviews for richer answers.