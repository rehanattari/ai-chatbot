"""
Multi-Model AI Chatbot with Complete Authentication
Supports OpenAI, Claude, Gemini, Grok, and more via OpenRouter
"""

import streamlit as st
import sqlite3
import json
import requests
import bcrypt
from datetime import datetime
from typing import List, Dict, Optional
import os
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================

# OpenRouter Public Endpoint
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Available AI Models
AVAILABLE_MODELS = {
    "Claude Sonnet 3.5": "anthropic/claude-3.5-sonnet",
    "GPT-4o": "openai/gpt-4o",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Gemini Pro 1.5": "google/gemini-pro-1.5",
    "Gemini Flash 1.5": "google/gemini-flash-1.5",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
    "Llama 3.1 405B": "meta-llama/llama-3.1-405b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
    "Command R+": "cohere/command-r-plus",
}

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Database path
DB_PATH = Path(__file__).parent / "database" / "conversations.db"
DB_PATH.parent.mkdir(exist_ok=True)

# ========================================
# DATABASE FUNCTIONS
# ========================================

def get_table_columns(cursor, table_name: str) -> list:
    """Get list of columns for a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [column[1] for column in cursor.fetchall()]

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Users table - create with all columns from start
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            avatar_url TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            api_key TEXT,
            is_active INTEGER DEFAULT 1,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
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
    
    # Check and add missing columns to existing tables
    columns = get_table_columns(cursor, "users")
    
    migrations = [
        ("email", "ALTER TABLE users ADD COLUMN email TEXT UNIQUE"),
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
                print(f"✅ Added {column_name} column to users table")
            except sqlite3.OperationalError as e:
                print(f"⚠️ Column {column_name} migration: {e}")
    
    conn.close()

# ========================================
# AUTHENTICATION FUNCTIONS
# ========================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False

def create_authenticated_user(username: str, email: str, display_name: str, password: str, avatar_seed: str = None) -> int:
    """Create a new user with authentication"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    if not avatar_seed:
        avatar_seed = username
    
    avatar_url = f"https://api.dicebear.com/7.x/avataaars/svg?seed={avatar_seed}"
    password_hash = hash_password(password)
    
    try:
        cursor.execute("""
            INSERT INTO users (username, email, display_name, avatar_url, password_hash, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (username, email, display_name, avatar_url, password_hash))
        
        user_id = cursor.lastrowid
        
        # Create default settings
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt)
            VALUES (?, ?)
        """, (user_id, "You are a helpful AI assistant."))
        
        conn.commit()
        return user_id
    except sqlite3.IntegrityError as e:
        print(f"User creation error: {e}")
        conn.rollback()
        return -1
    except Exception as e:
        print(f"Unexpected error: {e}")
        conn.rollback()
        return -2
    finally:
        conn.close()

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticate user with email and password"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, password_hash, api_key, is_active
            FROM users
            WHERE email = ? AND is_active = 1
        """, (email,))
        
        row = cursor.fetchone()
        
        if row and row[5]:  # Check if user exists and has password_hash
            if verify_password(password, row[5]):
                # Update last login
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
    except Exception as e:
        print(f"Authentication error: {e}")
    
    conn.close()
    return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, api_key
            FROM users
            WHERE user_id = ? AND is_active = 1
        """, (user_id,))
        
        row = cursor.fetchone()
        
        if row:
            return {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "display_name": row[3],
                "avatar_url": row[4],
                "api_key": row[5]
            }
    except Exception as e:
        print(f"Get user error: {e}")
    finally:
        conn.close()
    
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

def deactivate_user(user_id: int):
    """Deactivate user account"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# ========================================
# CONVERSATION FUNCTIONS
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
    
    # Update conversation timestamp
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
    
    # Check if settings exist
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
# AI API FUNCTIONS
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
    """Render login page"""
    st.markdown("""
        <style>
        .auth-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🤖 Sagoma AI")
        st.subheader("Sign In")
        
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
    """Render signup page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🤖 Sagoma AI")
        st.subheader("Create Account")
        
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
                    user_id = create_authenticated_user(username, email, display_name, password)
                    if user_id > 0:
                        st.success("Account created! Please sign in.")
                        st.session_state.show_signup = False
                        st.rerun()
                    elif user_id == -1:
                        st.error("Username or email already exists")
                    else:
                        st.error("An error occurred. Please try again.")
        
        with col_b:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()

# ========================================
# UI FUNCTIONS - MAIN APP
# ========================================

def render_header():
    """Render top header with user menu"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("🤖 Sagoma AI Chatbot")
    
    with col2:
        if st.session_state.authenticated and st.session_state.active_user:
            user = st.session_state.active_user
            
            # User menu
            st.markdown(f"""
                <div style="text-align: right; padding: 10px;">
                    <img src="{user['avatar_url']}" width="40" style="border-radius: 50%; vertical-align: middle;">
                    <span style="margin-left: 10px; font-weight: bold;">{user['display_name']}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Dropdown menu
            menu_col1, menu_col2, menu_col3 = st.columns(3)
            
            with menu_col1:
                if st.button("⚙️", key="settings_header", help="Settings"):
                    st.session_state.show_settings = True
                    st.rerun()
            
            with menu_col2:
                if st.button("🔑", key="api_key_header", help="API Key"):
                    st.session_state.show_api_key_modal = True
                    st.rerun()
            
            with menu_col3:
                if st.button("🚪", key="logout_header", help="Logout"):
                    st.session_state.authenticated = False
                    st.session_state.user_id = None
                    st.session_state.active_user = None
                    st.session_state.current_conversation = None
                    st.session_state.messages = []
                    st.rerun()

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
    
    with col2:
        if st.button("ℹ️ More Info", use_container_width=True):
            st.info("""
            **OpenRouter Pricing:**
            - GPT-4o Mini: ~$0.15 per 1M tokens
            - Claude 3.5 Sonnet: ~$3 per 1M tokens
            - Gemini Flash: ~$0.075 per 1M tokens
            
            **Example costs:**
            - 100 conversations ≈ $0.50 - $2.00
            - Most users spend <$5/month
            """)

def render_sidebar():
    """Render the sidebar with conversation history"""
    with st.sidebar:
        st.title("💬 Chat History")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Search conversations
        search_term = st.text_input("🔍 Search conversations", key="search_conv")
        
        # Display conversations
        if st.session_state.authenticated and st.session_state.active_user:
            conversations = get_user_conversations(st.session_state.active_user["user_id"])
            
            if search_term:
                conversations = [c for c in conversations if search_term.lower() in c["title"].lower()]
            
            if conversations:
                for conv in conversations:
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        title_display = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                        if st.button(
                            f"💬 {title_display}",
                            key=f"conv_{conv['conversation_id']}",
                            use_container_width=True
                        ):
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
            else:
                st.info("No conversations yet. Start a new chat!")

def render_settings():
    """Render settings modal"""
    if st.session_state.show_settings and st.session_state.active_user:
        user_id = st.session_state.active_user["user_id"]
        settings = get_user_settings(user_id)
        
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["Chat Behavior", "Account Security"])
        
        # Chat Behavior Tab
        with tabs[0]:
            st.subheader("🤖 Chatbot Customization")
            
            new_system_prompt = st.text_area(
                "System Prompt",
                value=settings.get("system_prompt", "You are a helpful AI assistant."),
                height=150,
                help="Define how the AI should behave"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=settings.get("temperature", 0.7),
                    step=0.1,
                    help="Controls randomness. Higher = more creative"
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
                    step=100,
                    help="Maximum response length"
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
                value=settings.get("stream_response", True),
                help="Show responses as they're generated"
            )
            
            if st.button("💾 Save Chat Settings", type="primary"):
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
        
        # Account Security Tab
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
                    # Verify current password
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
    """Render main chat interface"""
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
    
    # Model selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_display_names = list(AVAILABLE_MODELS.keys())
        model_values = list(AVAILABLE_MODELS.values())
        
        current_index = model_values.index(st.session_state.selected_model) if st.session_state.selected_model in model_values else 0
        
        selected_display = st.selectbox(
            "Select AI Model",
            options=model_display_names,
            index=current_index,
            key="model_selector"
        )
        st.session_state.selected_model = AVAILABLE_MODELS[selected_display]
    
    with col2:
        st.info(f"**Active**")
    
    st.divider()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Get settings
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
            
            # Prepare messages for API
            api_messages = [{"role": "system", "content": settings.get("system_prompt", "You are a helpful AI assistant.")}]
            api_messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])
            
            # Show loading spinner
            with st.spinner("🤖 Sagoma is processing request..."):
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
                
                # Trigger sidebar refresh to show new conversation
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
