"""
Multi-Model AI Chatbot with Complete Authentication and Custom Stitch Design
Supports OpenAI, Claude, Gemini, Grok, and more via OpenRouter
"""

import streamlit as st
import sqlite3
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional
import os
from pathlib import Path

# Try to import bcrypt, use hashlib as fallback
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    import hashlib
    BCRYPT_AVAILABLE = False

# ========================================
# CONFIGURATION
# ========================================

# OpenRouter Public Endpoint
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Available AI Models
AVAILABLE_MODELS = {
    "Gemini": "google/gemini-flash-1.5",
    "ChatGPT": "openai/gpt-4o-mini",
    "Claude": "anthropic/claude-3.5-sonnet",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat",
}

DEFAULT_MODEL = "google/gemini-flash-1.5"

# Database path
DB_PATH = Path(__file__).parent / "database" / "conversations.db"
DB_PATH.parent.mkdir(exist_ok=True)

# ========================================
# CUSTOM STITCH STYLING
# ========================================

def apply_stitch_design():
    """Apply custom Stitch design CSS to Streamlit"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
        
        /* Global Styles */
        * {
            font-family: 'Space Grotesk', sans-serif !important;
        }
        
        .stApp {
            background-color: #f7f7f7;
        }
        
        /* Dark mode */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #191919;
            }
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f3f4f6 !important;
            border-right: 1px solid #e5e7eb;
        }
        
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] {
                background-color: #151d2c !important;
                border-right: 1px solid #232f48;
            }
        }
        
        /* Sidebar Header */
        [data-testid="stSidebar"] h2 {
            color: #111827;
            font-size: 1.125rem;
            font-weight: 700;
            letter-spacing: -0.015em;
        }
        
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] h2 {
                color: white;
            }
        }
        
        /* New Chat Button */
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #000000 !important;
            color: white !important;
            border-radius: 0.5rem !important;
            font-weight: 700 !important;
            padding: 0.5rem 1rem !important;
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
        
        /* Conversation Items in Sidebar */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: transparent !important;
            color: #111827 !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem !important;
            text-align: left !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            margin-bottom: 0.25rem !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #e5e7eb !important;
        }
        
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] button[kind="secondary"] {
                color: white !important;
            }
            
            [data-testid="stSidebar"] button[kind="secondary"]:hover {
                background-color: #374151 !important;
            }
        }
        
        /* Main Header with Model Tabs */
        .model-tabs {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.25rem;
            background-color: #f3f4f6;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .model-tabs {
                background-color: #232f48;
            }
        }
        
        .model-tab {
            padding: 0.25rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            color: #6b7280;
            transition: all 0.2s;
        }
        
        .model-tab.active {
            background-color: #000000;
            color: white;
        }
        
        .model-tab:hover:not(.active) {
            background-color: #e5e7eb;
        }
        
        @media (prefers-color-scheme: dark) {
            .model-tab {
                color: #9ca3af;
            }
            
            .model-tab:hover:not(.active) {
                background-color: #374151;
            }
        }
        
        /* User Avatar in Header */
        .user-avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 9999px;
            background-size: cover;
            background-position: center;
            cursor: pointer;
        }
        
        /* Chat Messages */
        .stChatMessage {
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            border-radius: 0.5rem !important;
        }
        
        /* User Message */
        .stChatMessage[data-testid="user-message"] {
            background-color: #000000 !important;
            color: white !important;
            margin-left: auto !important;
            max-width: 70% !important;
            border-radius: 0.75rem 0.75rem 0.25rem 0.75rem !important;
        }
        
        /* AI Message */
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #e5e7eb !important;
            color: #111827 !important;
            max-width: 70% !important;
            border-radius: 0.75rem 0.75rem 0.75rem 0.25rem !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stChatMessage[data-testid="assistant-message"] {
                background-color: #232f48 !important;
                color: white !important;
            }
        }
        
        /* Chat Input */
        .stChatInputContainer {
            background-color: #e5e7eb !important;
            border-radius: 0.75rem !important;
            padding: 0.5rem !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stChatInputContainer {
                background-color: #232f48 !important;
            }
        }
        
        .stChatInput input {
            background-color: transparent !important;
            border: none !important;
            color: #111827 !important;
            font-size: 1rem !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stChatInput input {
                color: white !important;
            }
        }
        
        .stChatInput input::placeholder {
            color: #6b7280 !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stChatInput input::placeholder {
                color: #92a4c9 !important;
            }
        }
        
        /* Send Button */
        .stChatInputContainer button {
            background-color: #000000 !important;
            color: white !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }
        
        /* Loading Animation (Typing Indicator) */
        .typing-indicator {
            display: flex;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background-color: #e5e7eb;
            border-radius: 0.75rem;
            width: fit-content;
        }
        
        @media (prefers-color-scheme: dark) {
            .typing-indicator {
                background-color: #232f48;
            }
        }
        
        .typing-dot {
            width: 0.5rem;
            height: 0.5rem;
            background-color: #000000;
            border-radius: 9999px;
            animation: pulse 1.4s ease-in-out infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes pulse {
            0%, 60%, 100% {
                opacity: 0.3;
                transform: scale(0.8);
            }
            30% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Welcome Message */
        .welcome-message {
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
            padding: 1rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .welcome-message {
                color: #9ca3af;
            }
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #d1d5db;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #9ca3af;
        }
        
        @media (prefers-color-scheme: dark) {
            ::-webkit-scrollbar-thumb {
                background: #374151;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #4b5563;
            }
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# ========================================
# DATABASE FUNCTIONS (Keep existing functions)
# ========================================

def get_table_columns(cursor, table_name: str) -> list:
    """Get list of columns for a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [column[1] for column in cursor.fetchall()]

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        columns = get_table_columns(cursor, "users")
        migrations = [
            ("email", "ALTER TABLE users ADD COLUMN email TEXT"),
            ("password_hash", "ALTER TABLE users ADD COLUMN password_hash TEXT"),
            ("api_key", "ALTER TABLE users ADD COLUMN api_key TEXT"),
            ("is_active", "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"),
            ("last_login", "ALTER TABLE users ADD COLUMN last_login TIMESTAMP"),
        ]
        
        for column_name, sql in migrations:
            if column_name not in columns:
                try:
                    cursor.execute(sql)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
    else:
        cursor.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                avatar_url TEXT,
                email TEXT,
                password_hash TEXT,
                api_key TEXT,
                is_active INTEGER DEFAULT 1,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    
    # Conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    """)
    
    # Settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            user_id INTEGER PRIMARY KEY,
            system_prompt TEXT,
            temperature REAL DEFAULT 0.7,
            max_tokens INTEGER DEFAULT 2000,
            tone TEXT DEFAULT 'balanced',
            verbosity TEXT DEFAULT 'normal',
            stream_response INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    conn.commit()
    conn.close()

# ========================================
# AUTHENTICATION FUNCTIONS (Keep existing)
# ========================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt or fallback to sha256"""
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    else:
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    try:
        if BCRYPT_AVAILABLE and password_hash.startswith('$2'):
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        else:
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == password_hash
    except Exception:
        return False

def create_authenticated_user(username: str, email: str, display_name: str, password: str, avatar_seed: str = None) -> int:
    """Create a new user with authentication"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        if not avatar_seed:
            avatar_seed = username
        
        avatar_url = f"https://api.dicebear.com/7.x/avataaars/svg?seed={avatar_seed}"
        password_hash = hash_password(password)
        
        columns = get_table_columns(cursor, "users")
        
        if "email" in columns and "password_hash" in columns and "is_active" in columns:
            cursor.execute("""
                INSERT INTO users (username, email, display_name, avatar_url, password_hash, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (username, email, display_name, avatar_url, password_hash))
        else:
            cursor.execute("""
                INSERT INTO users (username, display_name, avatar_url)
                VALUES (?, ?, ?)
            """, (username, display_name, avatar_url))
        
        user_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt)
            VALUES (?, ?)
        """, (user_id, "You are a helpful AI assistant."))
        
        conn.commit()
        conn.close()
        return user_id
        
    except sqlite3.IntegrityError as e:
        if conn:
            conn.close()
        error_msg = str(e).lower()
        if "username" in error_msg:
            st.error("Username already exists")
        elif "email" in error_msg:
            st.error("Email already exists")
        return -1
        
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"Error creating user: {str(e)}")
        return -2

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticate user with email and password"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, password_hash, api_key, is_active
            FROM users
            WHERE email = ? AND is_active = 1
        """, (email,))
        
        row = cursor.fetchone()
        
        if row and row[5]:
            if verify_password(password, row[5]):
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?
                """, (row[0],))
                conn.commit()
                
                user = {
                    "user_id": row[0],
                    "username": row[1],
                    "email": row[2],
                    "display_name": row[3],
                    "avatar_url": row[4],
                    "api_key": row[6]
                }
                conn.close()
                return user
        
        conn.close()
        return None
        
    except Exception as e:
        st.error(f"Authentication error: {e}")
        if conn:
            conn.close()
        return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, api_key
            FROM users
            WHERE user_id = ? AND is_active = 1
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "display_name": row[3],
                "avatar_url": row[4],
                "api_key": row[5]
            }
        return None
        
    except Exception:
        return None

def update_user_api_key(user_id: int, api_key: str):
    """Update user's API key"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET api_key = ? WHERE user_id = ?", (api_key, user_id))
    conn.commit()
    conn.close()

def update_user_password(user_id: int, new_password: str):
    """Update user's password"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    password_hash = hash_password(new_password)
    cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))
    conn.commit()
    conn.close()

# ========================================
# CONVERSATION FUNCTIONS (Keep existing)
# ========================================

def get_user_conversations(user_id: int) -> List[Dict]:
    """Get all conversations for a user"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_id, title, model, created_at, updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
    """, (user_id,))
    
    conversations = [
        {
            "conversation_id": row[0],
            "title": row[1],
            "model": row[2],
            "created_at": row[3],
            "updated_at": row[4]
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return conversations

def create_conversation(user_id: int, title: str, model: str) -> int:
    """Create a new conversation"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (user_id, title, model)
        VALUES (?, ?, ?)
    """, (user_id, title, model))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: int) -> List[Dict]:
    """Get all messages in a conversation"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """, (conversation_id,))
    
    messages = [
        {"role": row[0], "content": row[1], "timestamp": row[2]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return messages

def add_message(conversation_id: int, role: str, content: str):
    """Add a message to a conversation"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
    """, (conversation_id, role, content))
    
    cursor.execute("""
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE conversation_id = ?
    """, (conversation_id,))
    
    conn.commit()
    conn.close()

def get_user_settings(user_id: int) -> Dict:
    """Get user settings"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT system_prompt, temperature, max_tokens, tone, verbosity, stream_response
        FROM settings
        WHERE user_id = ?
    """, (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "system_prompt": row[0],
            "temperature": row[1],
            "max_tokens": row[2],
            "tone": row[3],
            "verbosity": row[4],
            "stream_response": bool(row[5])
        }
    return {
        "system_prompt": "You are a helpful AI assistant.",
        "temperature": 0.7,
        "max_tokens": 2000,
        "tone": "balanced",
        "verbosity": "normal",
        "stream_response": True
    }

def update_user_settings(user_id: int, settings: Dict):
    """Update user settings"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_id FROM settings WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone()
    
    if exists:
        cursor.execute("""
            UPDATE settings
            SET system_prompt = ?,
                temperature = ?,
                max_tokens = ?,
                tone = ?,
                verbosity = ?,
                stream_response = ?
            WHERE user_id = ?
        """, (
            settings["system_prompt"],
            settings["temperature"],
            settings["max_tokens"],
            settings["tone"],
            settings["verbosity"],
            int(settings["stream_response"]),
            user_id
        ))
    else:
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt, temperature, max_tokens, tone, verbosity, stream_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            settings["system_prompt"],
            settings["temperature"],
            settings["max_tokens"],
            settings["tone"],
            settings["verbosity"],
            int(settings["stream_response"])
        ))
    
    conn.commit()
    conn.close()

# ========================================
# AI API FUNCTIONS (Keep existing)
# ========================================

def call_ai_model(messages: List[Dict], model: str, settings: Dict, api_key: str, stream: bool = True):
    """Call AI model via OpenRouter"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/ai-chatbot",
        "X-Title": "Sagoma AI Chatbot"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": settings.get("temperature", 0.7),
        "max_tokens": settings.get("max_tokens", 2000),
        "stream": stream
    }
    
    try:
        response = requests.post(
            OPENROUTER_ENDPOINT,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=300
        )
        
        response.raise_for_status()
        
        if stream:
            return response
        else:
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def stream_ai_response(messages: List[Dict], model: str, settings: Dict, api_key: str):
    """Stream AI response token by token"""
    response = call_ai_model(messages, model, settings, api_key, stream=True)
    
    if response is None:
        return
    
    full_response = ""
    
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            if line_text.startswith("data: "):
                data_str = line_text[6:]
                
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            chunk = delta["content"]
                            full_response += chunk
                            yield chunk
                except json.JSONDecodeError:
                    continue
    
    return full_response

# ========================================
# UI FUNCTIONS - AUTHENTICATION
# ========================================

def render_login():
    """Render login page with Stitch styling"""
    st.markdown("""
        <div style="max-width: 400px; margin: 100px auto; padding: 40px; border-radius: 10px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">🤖 Sagoma AI</h1>
            <h2 style="text-align: center; font-size: 1.5rem; color: #6b7280; margin-bottom: 2rem;">Sign In</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        email = st.text_input("Email", placeholder="your@email.com", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔐 Sign In", type="primary", use_container_width=True):
                if email and password:
                    user = authenticate_user(email, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user["user_id"]
                        st.session_state.active_user = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.error("Please enter both email and password")
        
        with col_b:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()

def render_signup():
    """Render signup page with Stitch styling"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">🤖 Sagoma AI</h1>
                <h2 style="font-size: 1.5rem; color: #6b7280;">Create Account</h2>
            </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="johndoe", key="signup_username")
        email = st.text_input("Email", placeholder="your@email.com", key="signup_email")
        display_name = st.text_input("Display Name", placeholder="John Doe", key="signup_display")
        password = st.text_input("Password", type="password", key="signup_password")
        password_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("✅ Create Account", type="primary", use_container_width=True):
                if not all([username, email, display_name, password, password_confirm]):
                    st.error("Please fill all fields")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif "@" not in email or "." not in email:
                    st.error("Please enter a valid email address")
                else:
                    with st.spinner("Creating account..."):
                        user_id = create_authenticated_user(username, email, display_name, password)
                        if user_id > 0:
                            st.success("✅ Account created! Please sign in.")
                            st.balloons()
                            st.session_state.show_signup = False
                            st.rerun()
        
        with col_b:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()

# ========================================
# UI FUNCTIONS - MAIN APP
# ========================================

def render_header():
    """Render top header with model tabs and user avatar"""
    # Model selector tabs
    st.markdown('<div class="model-tabs">', unsafe_allow_html=True)
    
    cols = st.columns(len(AVAILABLE_MODELS) + 2)
    
    with cols[0]:
        st.write("")  # Spacer
    
    for idx, (model_name, model_id) in enumerate(AVAILABLE_MODELS.items(), 1):
        with cols[idx]:
            active_class = "active" if st.session_state.selected_model == model_id else ""
            if st.button(model_name, key=f"model_{model_name}", use_container_width=True):
                st.session_state.selected_model = model_id
                st.rerun()
    
    with cols[-1]:
        if st.session_state.authenticated and st.session_state.active_user:
            user = st.session_state.active_user
            
            # User avatar with dropdown
            with st.popover("👤", use_container_width=False):
                st.markdown(f"**{user['display_name']}**")
                st.markdown(f"*{user.get('email', 'N/A')}*")
                st.divider()
                
                if st.button("⚙️ Settings", use_container_width=True):
                    st.session_state.show_settings = True
                    st.rerun()
                
                if st.button("🔑 API Key", use_container_width=True):
                    st.session_state.show_api_key_modal = True
                    st.rerun()
                
                if st.button("🚪 Logout", use_container_width=True):
                    st.session_state.authenticated = False
                    st.session_state.user_id = None
                    st.session_state.active_user = None
                    st.session_state.current_conversation = None
                    st.session_state.messages = []
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_api_key_setup():
    """Render API key setup screen"""
    st.title("🔑 API Key Setup Required")
    
    st.markdown("""
    ### Welcome to Sagoma AI Chatbot!
    
    To use this chatbot, you need an **OpenRouter API key** (it's free to get started).
    
    #### How to get your API key:
    
    1. Go to [OpenRouter.ai](https://openrouter.ai/)
    2. Click **"Sign In"** (use Google, GitHub, or email)
    3. Click on your profile → **"Keys"**
    4. Click **"Create Key"**
    5. Copy your API key
    6. Paste it below
    
    #### Why OpenRouter?
    - ✅ Access to 100+ AI models (GPT-4, Claude, Gemini, etc.)
    - ✅ Pay-as-you-go pricing (no subscription)
    - ✅ Free credits to start ($1-5 depending on promotions)
    - ✅ Simple, unified API
    
    """)
    
    st.divider()
    
    api_key = st.text_input(
        "Enter your OpenRouter API Key:",
        type="password",
        placeholder="sk-or-v1-..."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("💾 Save Key", type="primary", use_container_width=True):
            if api_key and api_key.startswith("sk-or-"):
                update_user_api_key(st.session_state.active_user["user_id"], api_key)
                st.session_state.active_user["api_key"] = api_key
                st.session_state.show_api_key_modal = False
                st.success("API key saved!")
                st.rerun()
            else:
                st.error("Please enter a valid OpenRouter API key (starts with 'sk-or-')")

def render_sidebar():
    """Render the sidebar with conversation history"""
    with st.sidebar:
        # Header with logo
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                <svg width="24" height="24" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M24 4C25.7818 14.2173 33.7827 22.2182 44 24C33.7827 25.7818 25.7818 33.7827 24 44C22.2182 33.7827 14.2173 25.7818 4 24C14.2173 22.2182 22.2182 14.2173 24 4Z" fill="currentColor"/>
                </svg>
                <h2 style="font-size: 1.125rem; font-weight: 700; margin: 0;">AI Chat</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # New chat button
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Search conversations
        search_term = st.text_input("🔍 Search", key="search_conv", label_visibility="collapsed")
        
        # Display conversations
        if st.session_state.authenticated and st.session_state.active_user:
            conversations = get_user_conversations(st.session_state.active_user["user_id"])
            
            if search_term:
                conversations = [c for c in conversations if search_term.lower() in c["title"].lower()]
            
            # Group conversations by date
            today = []
            yesterday = []
            older = []
            
            from datetime import datetime, timedelta
            now = datetime.now()
            
            for conv in conversations:
                conv_date = datetime.fromisoformat(conv["updated_at"])
                if conv_date.date() == now.date():
                    today.append(conv)
                elif conv_date.date() == (now - timedelta(days=1)).date():
                    yesterday.append(conv)
                else:
                    older.append(conv)
            
            # Display Today
            if today:
                st.markdown('<p style="color: #6b7280; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; padding: 0.5rem 0;">Today</p>', unsafe_allow_html=True)
                for conv in today:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        title_display = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                        if st.button(f"💬 {title_display}", key=f"conv_{conv['conversation_id']}", use_container_width=True):
                            st.session_state.current_conversation = conv
                            st.session_state.messages = get_conversation_messages(conv["conversation_id"])
                            st.session_state.selected_model = conv["model"]
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['conversation_id']}"):
                            delete_conversation(conv["conversation_id"])
                            if st.session_state.current_conversation and \
                               st.session_state.current_conversation["conversation_id"] == conv["conversation_id"]:
                                st.session_state.current_conversation = None
                                st.session_state.messages = []
                            st.rerun()
            
            # Display Yesterday
            if yesterday:
                st.markdown('<p style="color: #6b7280; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; padding: 0.5rem 0;">Yesterday</p>', unsafe_allow_html=True)
                for conv in yesterday:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        title_display = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                        if st.button(f"💬 {title_display}", key=f"conv_{conv['conversation_id']}", use_container_width=True):
                            st.session_state.current_conversation = conv
                            st.session_state.messages = get_conversation_messages(conv["conversation_id"])
                            st.session_state.selected_model = conv["model"]
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['conversation_id']}"):
                            delete_conversation(conv["conversation_id"])
                            if st.session_state.current_conversation and \
                               st.session_state.current_conversation["conversation_id"] == conv["conversation_id"]:
                                st.session_state.current_conversation = None
                                st.session_state.messages = []
                            st.rerun()
            
            # Display Older
            if older:
                st.markdown('<p style="color: #6b7280; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; padding: 0.5rem 0;">Older</p>', unsafe_allow_html=True)
                for conv in older:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        title_display = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                        if st.button(f"💬 {title_display}", key=f"conv_{conv['conversation_id']}", use_container_width=True):
                            st.session_state.current_conversation = conv
                            st.session_state.messages = get_conversation_messages(conv["conversation_id"])
                            st.session_state.selected_model = conv["model"]
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['conversation_id']}"):
                            delete_conversation(conv["conversation_id"])
                            if st.session_state.current_conversation and \
                               st.session_state.current_conversation["conversation_id"] == conv["conversation_id"]:
                                st.session_state.current_conversation = None
                                st.session_state.messages = []
                            st.rerun()
            
            if not (today or yesterday or older):
                st.markdown('<p class="welcome-message">No conversations yet. Start a new chat!</p>', unsafe_allow_html=True)

def render_settings():
    """Render settings modal"""
    if st.session_state.show_settings and st.session_state.active_user:
        user_id = st.session_state.active_user["user_id"]
        settings = get_user_settings(user_id)
        
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["Chat Behavior", "Account Security"])
        
        with tabs[0]:
            st.subheader("🤖 Chatbot Customization")
            
            new_system_prompt = st.text_area(
                "System Prompt",
                value=settings.get("system_prompt", "You are a helpful AI assistant."),
                height=150
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=settings.get("temperature", 0.7),
                    step=0.1
                )
                
                new_tone = st.selectbox(
                    "Tone",
                    options=["professional", "casual", "friendly", "technical", "balanced"],
                    index=["professional", "casual", "friendly", "technical", "balanced"].index(
                        settings.get("tone", "balanced")
                    )
                )
            
            with col2:
                new_max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=4000,
                    value=settings.get("max_tokens", 2000),
                    step=100
                )
                
                new_verbosity = st.selectbox(
                    "Verbosity",
                    options=["concise", "normal", "detailed"],
                    index=["concise", "normal", "detailed"].index(
                        settings.get("verbosity", "normal")
                    )
                )
            
            new_stream = st.checkbox(
                "Stream responses",
                value=settings.get("stream_response", True)
            )
            
            if st.button("💾 Save Settings", type="primary"):
                update_user_settings(user_id, {
                    "system_prompt": new_system_prompt,
                    "temperature": new_temperature,
                    "max_tokens": new_max_tokens,
                    "tone": new_tone,
                    "verbosity": new_verbosity,
                    "stream_response": new_stream
                })
                st.success("Settings saved!")
                st.rerun()
        
        with tabs[1]:
            st.subheader("🔒 Account Security")
            
            user = st.session_state.active_user
            
            st.write(f"**Email:** {user.get('email', 'N/A')}")
            st.write(f"**Username:** {user['username']}")
            
            st.divider()
            
            st.subheader("Change Password")
            
            current_password = st.text_input("Current Password", type="password", key="current_pwd")
            new_password = st.text_input("New Password", type="password", key="new_pwd")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pwd")
            
            if st.button("🔐 Update Password", type="primary"):
                if not all([current_password, new_password, confirm_password]):
                    st.error("Please fill all fields")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if authenticate_user(user['email'], current_password):
                        update_user_password(user_id, new_password)
                        st.success("Password updated successfully!")
                    else:
                        st.error("Current password is incorrect")
        
        st.divider()
        
        if st.button("✖️ Close Settings"):
            st.session_state.show_settings = False
            st.rerun()

def render_chat():
    """Render main chat interface with Stitch design"""
    # Check if API key is configured
    if not st.session_state.active_user.get("api_key"):
        if st.session_state.show_api_key_modal:
            render_api_key_setup()
        else:
            st.session_state.show_api_key_modal = True
            st.rerun()
        return
    
    if st.session_state.show_settings:
        render_settings()
        return
    
    if st.session_state.show_api_key_modal:
        render_api_key_setup()
        return
    
    # Welcome message if no messages
    if not st.session_state.messages:
        st.markdown('<p class="welcome-message">Welcome! Ask me anything.</p>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        settings = get_user_settings(st.session_state.active_user["user_id"])
        api_key = st.session_state.active_user["api_key"]
        
        # Create conversation immediately on first message
        if st.session_state.current_conversation is None:
            title = prompt[:50] if len(prompt) < 50 else prompt[:47] + "..."
            conv_id = create_conversation(
                st.session_state.active_user["user_id"],
                title,
                st.session_state.selected_model
            )
            st.session_state.current_conversation = {
                "conversation_id": conv_id,
                "title": title,
                "model": st.session_state.selected_model
            }
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        add_message(st.session_state.current_conversation["conversation_id"], "user", prompt)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response with loading indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show typing indicator
            message_placeholder.markdown("""
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            """, unsafe_allow_html=True)
            
            # Prepare messages for API
            api_messages = [{"role": "system", "content": settings.get("system_prompt", "You are a helpful AI assistant.")}]
            api_messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])
            
            if settings.get("stream_response", True):
                full_response = ""
                for chunk in stream_ai_response(
                    api_messages,
                    st.session_state.selected_model,
                    settings,
                    api_key
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                full_response = call_ai_model(
                    api_messages,
                    st.session_state.selected_model,
                    settings,
                    api_key,
                    stream=False
                )
                if full_response:
                    message_placeholder.markdown(full_response)
            
            # Save assistant message
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                add_message(st.session_state.current_conversation["conversation_id"], "assistant", full_response)
                st.rerun()

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    st.set_page_config(
        page_title="Sagoma AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom Stitch design
    apply_stitch_design()
    
    # Initialize database
    init_database()
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    if "active_user" not in st.session_state:
        st.session_state.active_user = None
    
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False
    
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    if "show_api_key_modal" not in st.session_state:
        st.session_state.show_api_key_modal = False
    
    # Authentication flow
    if not st.session_state.authenticated:
        if st.session_state.show_signup:
            render_signup()
        else:
            render_login()
    else:
        # Load user data if needed
        if st.session_state.active_user is None and st.session_state.user_id:
            st.session_state.active_user = get_user_by_id(st.session_state.user_id)
        
        # Render main app
        render_header()
        render_sidebar()
        render_chat()

if __name__ == "__main__":
    main()
