"""
Multi-Model AI Chatbot with Complete Authentication
Supports OpenAI, Claude, Gemini, Grok, and more via OpenRouter
"""

import streamlit as st
import sqlite3
import json
import requests
from datetime import datetime, timedelta
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
# THEME MANAGEMENT
# ========================================

def apply_theme():
    """Apply custom CSS based on theme"""
    # Get current theme
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        st.markdown("""
            <style>
            /* Dark mode styles */
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            [data-testid="stSidebar"] {
                background-color: #262730;
            }
            .stTextInput input, .stTextArea textarea, .stSelectbox select {
                background-color: #262730;
                color: #fafafa;
            }
            .user-avatar-dropdown {
                background-color: #262730;
                color: #fafafa;
                border: 1px solid #464646;
            }
            .conversation-group-header {
                color: #a0a0a0;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            /* Light mode styles */
            .stApp {
                background-color: #ffffff;
                color: #31333F;
            }
            [data-testid="stSidebar"] {
                background-color: #f0f2f6;
            }
            .user-avatar-dropdown {
                background-color: #ffffff;
                color: #31333F;
                border: 1px solid #e0e0e0;
            }
            .conversation-group-header {
                color: #6e6e6e;
            }
            </style>
        """, unsafe_allow_html=True)
    
    # Common styles
    st.markdown("""
        <style>
        /* User avatar styling */
        .user-avatar-container {
            position: relative;
            display: inline-block;
        }
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            object-fit: cover;
        }
        .user-avatar-dropdown {
            position: absolute;
            top: 50px;
            right: 0;
            min-width: 200px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 0.5rem;
            z-index: 1000;
        }
        .dropdown-item {
            padding: 0.75rem 1rem;
            cursor: pointer;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: background-color 0.2s;
        }
        .dropdown-item:hover {
            background-color: rgba(0,0,0,0.05);
        }
        .conversation-group-header {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.5rem 0;
            margin-top: 1rem;
        }
        .theme-toggle {
            padding: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

# ========================================
# DATABASE FUNCTIONS (Keep existing)
# ========================================

def get_table_columns(cursor, table_name: str) -> list:
    """Get list of columns for a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [column[1] for column in cursor.fetchall()]

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
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
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    else:
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    try:
        if BCRYPT_AVAILABLE and password_hash.startswith('$2'):
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        else:
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == password_hash
    except Exception:
        return False

def create_authenticated_user(username: str, email: str, display_name: str, password: str, avatar_seed: str = None) -> int:
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
        
    except Exception:
        if conn:
            conn.close()
        return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
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
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET api_key = ? WHERE user_id = ?", (api_key, user_id))
    conn.commit()
    conn.close()

def update_user_password(user_id: int, new_password: str):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    password_hash = hash_password(new_password)
    cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))
    conn.commit()
    conn.close()

# ========================================
# CONVERSATION FUNCTIONS
# ========================================

def get_user_conversations(user_id: int) -> List[Dict]:
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

def group_conversations_by_date(conversations: List[Dict]) -> Dict[str, List[Dict]]:
    """Group conversations by Today, Yesterday, and specific dates"""
    now = datetime.now()
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()
    
    groups = {
        "Today": [],
        "Yesterday": [],
    }
    older_dates = {}
    
    for conv in conversations:
        try:
            conv_datetime = datetime.fromisoformat(conv["updated_at"])
            conv_date = conv_datetime.date()
            
            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            else:
                # Format date as "Month Day, Year" (e.g., "January 15, 2024")
                date_str = conv_datetime.strftime("%B %d, %Y")
                if date_str not in older_dates:
                    older_dates[date_str] = []
                older_dates[date_str].append(conv)
        except (ValueError, TypeError):
            # If date parsing fails, put in "Older"
            if "Older" not in older_dates:
                older_dates["Older"] = []
            older_dates["Older"].append(conv)
    
    # Merge older dates into groups (sorted by date)
    for date_str in sorted(older_dates.keys(), reverse=True):
        groups[date_str] = older_dates[date_str]
    
    return groups

def create_conversation(user_id: int, title: str, model: str) -> int:
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
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: int) -> List[Dict]:
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
# UI FUNCTIONS - AUTHENTICATION (Keep existing)
# ========================================

def render_login():
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
    """Render top header with theme toggle and user avatar dropdown"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🤖 Sagoma AI Chatbot")
    
    with col2:
        # Theme toggle
        current_theme = st.session_state.get('theme', 'light')
        theme_icon = "🌙" if current_theme == 'light' else "☀️"
        theme_label = "Dark Mode" if current_theme == 'light' else "Light Mode"
        
        if st.button(f"{theme_icon} {theme_label}", key="theme_toggle"):
            st.session_state.theme = 'dark' if current_theme == 'light' else 'light'
            st.rerun()
    
    with col3:
        if st.session_state.authenticated and st.session_state.active_user:
            user = st.session_state.active_user
            
            # User avatar dropdown using popover
            with st.popover("👤 " + user['display_name'][:10], use_container_width=False):
                st.markdown(f"### {user['display_name']}")
                st.markdown(f"*{user.get('email', 'N/A')}*")
                st.divider()
                
                # My Profile
                if st.button("👤 My Profile", key="profile_btn", use_container_width=True):
                    st.session_state.show_profile = True
                    st.rerun()
                
                # Settings
                if st.button("⚙️ Settings", key="settings_btn_dropdown", use_container_width=True):
                    st.session_state.show_settings = True
                    st.rerun()
                
                # API Key
                if st.button("🔑 API Key", key="api_key_btn_dropdown", use_container_width=True):
                    st.session_state.show_api_key_modal = True
                    st.rerun()
                
                st.divider()
                
                # Logout
                if st.button("🚪 Logout", key="logout_btn", use_container_width=True, type="primary"):
                    st.session_state.authenticated = False
                    st.session_state.user_id = None
                    st.session_state.active_user = None
                    st.session_state.current_conversation = None
                    st.session_state.messages = []
                    st.rerun()

def render_profile():
    """Render user profile page"""
    if st.session_state.show_profile and st.session_state.active_user:
        user = st.session_state.active_user
        
        st.title("👤 My Profile")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(user["avatar_url"], width=150)
        
        with col2:
            st.markdown(f"### {user['display_name']}")
            st.markdown(f"**Username:** {user['username']}")
            st.markdown(f"**Email:** {user.get('email', 'N/A')}")
            st.markdown(f"**User ID:** {user['user_id']}")
        
        st.divider()
        
        st.info("💡 To update your profile information, please contact support or create a new account.")
        
        st.divider()
        
        if st.button("← Back to Chat", type="primary"):
            st.session_state.show_profile = False
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

def render_sidebar():
    """Render the sidebar with grouped conversation history"""
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
        
        # Display grouped conversations
        if st.session_state.authenticated and st.session_state.active_user:
            conversations = get_user_conversations(st.session_state.active_user["user_id"])
            
            if search_term:
                conversations = [c for c in conversations if search_term.lower() in c["title"].lower()]
            
            if conversations:
                # Group conversations by date
                grouped_convs = group_conversations_by_date(conversations)
                
                for group_name, group_convs in grouped_convs.items():
                    if group_convs:  # Only show non-empty groups
                        # Group header
                        st.markdown(f'<p class="conversation-group-header">{group_name}</p>', unsafe_allow_html=True)
                        
                        for conv in group_convs:
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
    """Render main chat interface"""
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
    
    if st.session_state.get('show_profile', False):
        render_profile()
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
        settings = get_user_settings(st.session_state.active_user["user_id"])
        api_key = st.session_state.active_user["api_key"]
        
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
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        add_message(st.session_state.current_conversation["conversation_id"], "user", prompt)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            api_messages = [{"role": "system", "content": settings.get("system_prompt", "You are a helpful AI assistant.")}]
            api_messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])
            
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
    
    # Apply theme
    apply_theme()
    
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
    
    if "show_profile" not in st.session_state:
        st.session_state.show_profile = False
    
    if "theme" not in st.session_state:
        st.session_state.theme = 'light'
    
    # Authentication flow
    if not st.session_state.authenticated:
        if st.session_state.show_signup:
            render_signup()
        else:
            render_login()
    else:
        if st.session_state.active_user is None and st.session_state.user_id:
            st.session_state.active_user = get_user_by_id(st.session_state.user_id)
        
        render_header()
        render_sidebar()
        render_chat()

if __name__ == "__main__":
    main()
