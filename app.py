import os
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import re
from typing import List, Dict, Any
import json

# Load environment variables
load_dotenv()

# Fetch values
DATABASE_URL = os.getenv("DATABASE_URL1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY1")
GROQ_MODEL = os.getenv("GROQ_MODEL1")
app = FastAPI(title="ROG Xbox Ally Chatbot", version="1.0.0")

# # Configuration
# DATABASE_URL = DATABASE_URL1
# GROQ_API_KEY = GROQ_API_KEY1
# GROQ_MODEL = GROQ_MODEL1

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

def search_similar_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar chunks using vector similarity from existing items_xbox table"""
    # Generate query embedding
    query_embedding = model.encode([query])[0].tolist()
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # First, let's check the table structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'items_xbox'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            print(f"Table structure: {columns}")
            
            # Check if embedding column exists and its type
            cursor.execute("""
                SELECT embedding FROM items_xbox LIMIT 1
            """)
            
            sample = cursor.fetchone()
            if sample:
                print(f"Sample embedding type: {type(sample['embedding'])}")
                print(f"Sample embedding: {sample['embedding'][:5] if hasattr(sample['embedding'], '__getitem__') else 'Not iterable'}")
            
            # Try different query approaches based on the actual table structure
            try:
                # Try the original query first
                cursor.execute("""
                    SELECT 
                        text,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM items_xbox
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, top_k))
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
            except Exception as e:
                print(f"First query failed: {e}")
                
                # Try alternative approach - cast the array to vector
                try:
                    cursor.execute("""
                        SELECT 
                            text,
                            1 - (embedding::vector <=> %s::vector) as similarity
                        FROM items_xbox
                        ORDER BY embedding::vector <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, top_k))
                    
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                    
                except Exception as e2:
                    print(f"Second query failed: {e2}")
                    
                    # Last resort - simple text search
                    cursor.execute("""
                        SELECT 
                            text,
                            0.5 as similarity
                        FROM items_xbox
                        WHERE text ILIKE %s
                        LIMIT %s
                    """, (f'%{query}%', top_k))
                    
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
            
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    finally:
        conn.close()

def generate_answer(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """Generate answer using Groq LLM"""
    try:
        # Build context from relevant chunks
        context = "\n\n".join([f"Source {i+1}: {chunk['text']}" 
                              for i, chunk in enumerate(relevant_chunks)])
        
        # Create prompt
        prompt = f"""You are a helpful assistant answering questions about the Xbox ROG Ally handheld device. 
        Use the following information to answer the user's question. If you don't know the answer based on the provided information, say so.

        Context:
        {context}

        User Question: {query}

        Answer:"""
        
        # Call Groq API
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"LLM error: {e}")
        return f"Sorry, I encountered an error generating the answer: {e}"

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for RAG-based responses using existing data"""
    try:
        # Search for relevant chunks from existing items_xbox table
        relevant_chunks = search_similar_chunks(request.message)
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I don't have enough information to answer your question. The Xbox ROG Ally data needs to be available in the database.",
                sources=[]
            )
        
        # Generate answer using LLM
        answer = generate_answer(request.message, relevant_chunks)
        
        # Format sources
        sources = [
            {
                "content": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "similarity": round(chunk["similarity"], 3)
            }
            for chunk in relevant_chunks
        ]
        
        return ChatResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the chatbot frontend"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROG Xbox Ally Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.8;
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .chat-messages {
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .user-message {
            background: linear-gradient(135deg, #00ff88, #00ccff);
            color: #000;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            font-size: 16px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .chat-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        .send-btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #00ff88, #00ccff);
            color: #000;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
        }
        
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .sources {
            margin-top: 10px;
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .source-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 3px solid #00ff88;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            opacity: 0.7;
        }
        
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ROG Xbox Ally Chatbot</h1>
            <p>Ask me anything about the Xbox ROG Ally handheld device!</p>
        </div>
        
        <div class="status">
            ✅ Ready to chat! Using existing Xbox ROG Ally data from database.
        </div>
        
        <div class="chat-container">
            <div id="chatMessages" class="chat-messages">
                <div class="message bot-message">
                    👋 Hello! I'm your Xbox ROG Ally assistant. I'm ready to answer your questions using the existing data. Ask me anything about the device!
                </div>
            </div>
            
            <div class="input-container">
                <input type="text" id="chatInput" class="chat-input" placeholder="Ask about the ROG Ally..." onkeypress="handleKeyPress(event)">
                <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
            </div>
            
            <div id="loading" class="loading">
                🤔 Thinking...
            </div>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('sendBtn').disabled = true;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Add bot response
                    addMessage(result.answer, 'bot');
                    
                    // Add sources if available
                    if (result.sources && result.sources.length > 0) {
                        const sourcesDiv = document.createElement('div');
                        sourcesDiv.className = 'sources';
                        sourcesDiv.innerHTML = '<strong>📚 Sources:</strong>';
                        
                        result.sources.forEach((source, index) => {
                            const sourceItem = document.createElement('div');
                            sourceItem.className = 'source-item';
                            sourceItem.innerHTML = `
                                <strong>Source ${index + 1}</strong> (Similarity: ${source.similarity})<br>
                                ${source.content}
                            `;
                            sourcesDiv.appendChild(sourceItem);
                        });
                        
                        document.getElementById('chatMessages').appendChild(sourcesDiv);
                    }
                } else {
                    addMessage(`❌ Error: ${result.detail}`, 'bot');
                }
            } catch (error) {
                addMessage(`❌ Error: ${error.message}`, 'bot');
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('sendBtn').disabled = false;
                input.focus();
            }
        }
        
        function addMessage(text, sender) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080) 