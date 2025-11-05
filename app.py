"""
Multi-Model AI Chatbot with File Upload, Image Generation, and Voice Chat
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
import secrets
import string
import base64
from io import BytesIO

# Try to import optional dependencies
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    import hashlib
    BCRYPT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ========================================
# CONFIGURATION
# ========================================

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_API_KEY = "sk-or-v1-YOUR_API_KEY_HERE"

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@sagoma.ai")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@2024")

SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price": 0,
        "features": [
            "Access to basic models",
            "Text chat only",
            "Limited file uploads (5MB)",
            "Standard response time"
        ]
    },
    "pro": {
        "name": "Pro",
        "price": 10,
        "features": [
            "Access to all premium models",
            "File uploads up to 50MB",
            "Image generation (10/day)",
            "Voice chat enabled",
            "Priority support"
        ]
    },
    "plus": {
        "name": "Plus",
        "price": 20,
        "features": [
            "Everything in Pro",
            "Unlimited file uploads",
            "Unlimited image generation",
            "Voice chat with custom voices",
            "API access"
        ]
    }
}

AVAILABLE_MODELS = {
    "Claude Sonnet 3.5": "anthropic/claude-3.5-sonnet",
    "GPT-4o": "openai/gpt-4o",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Gemini Pro 1.5": "google/gemini-pro-1.5",
    "Gemini Flash 1.5": "google/gemini-flash-1.5",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat",
}

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
IMAGE_MODEL = "replicate/black-forest-labs/flux-1.1-pro"

SECURITY_QUESTIONS = [
    "What is your mother's maiden name?",
    "What was the name of your first pet?",
    "What city were you born in?",
    "What is your favorite book?",
    "What was your childhood nickname?",
    "What is the name of your elementary school?",
]

DB_PATH = Path(__file__).parent / "database" / "conversations.db"
DB_PATH.parent.mkdir(exist_ok=True)

# ========================================
# FILE PROCESSING
# ========================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_AVAILABLE:
        return "[PDF processing unavailable]"
    try:
        pdf_file = BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"[Error: {str(e)}]"

def process_image_file(file_bytes: bytes) -> str:
    if not PIL_AVAILABLE:
        return None
    try:
        image = Image.open(BytesIO(file_bytes))
        max_size = (800, 800)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return None

# ========================================
# IMAGE GENERATION
# ========================================

def generate_image(prompt: str) -> Optional[str]:
    headers = {
        "customerId": "cus_THK5Sv5YGMPz67",
        "Content-Type": "application/json",
        "Authorization": "Bearer xxx"
    }
    
    payload = {
        "model": IMAGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Generate an image: {prompt}"}
        ]
    }
    
    try:
        response = requests.post(
            "https://llm.blackbox.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            return content
        return None
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return None

# ========================================
# SECURITY
# ========================================

def generate_recovery_code() -> str:
    characters = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(characters) for _ in range(16))

def hash_data(data: str) -> str:
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(data.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    else:
        import hashlib
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

def verify_hashed_data(data: str, hashed: str) -> bool:
    try:
        if BCRYPT_AVAILABLE and hashed.startswith('$2'):
            return bcrypt.checkpw(data.encode('utf-8'), hashed.encode('utf-8'))
        else:
            import hashlib
            return hashlib.sha256(data.encode('utf-8')).hexdigest() == hashed
    except Exception:
        return False

# ========================================
# THEME
# ========================================

def apply_theme():
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            [data-testid="stSidebar"] { background-color: #262730; }
            .conversation-group-header { color: #a0a0a0; }
            .subscription-card { background-color: #262730; border: 2px solid #464646; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #ffffff; color: #31333F; }
            [data-testid="stSidebar"] { background-color: #f0f2f6; }
            .conversation-group-header { color: #6e6e6e; }
            .subscription-card { background-color: #f8f9fa; border: 2px solid #e0e0e0; }
            .subscription-card.recommended { border-color: #4CAF50; background-color: #f0fdf4; }
            </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .conversation-group-header {
            font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
            padding: 0.5rem 0; margin-top: 1rem;
        }
        .subscription-card { padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
        .admin-badge {
            background-color: #ef4444; color: white;
            padding: 0.25rem 0.5rem; border-radius: 4px;
            font-size: 0.75rem; font-weight: 700;
        }
        .feature-badge {
            background-color: #10b981; color: white;
            padding: 0.25rem 0.5rem; border-radius: 4px;
            font-size: 0.75rem; font-weight: 700; margin-left: 0.5rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# ========================================
# DATABASE (SHORTENED - keeping essential parts)
# ========================================

def get_table_columns(cursor, table_name: str) -> list:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [column[1] for column in cursor.fetchall()]

def init_database():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Migrate users table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if cursor.fetchone():
        columns = get_table_columns(cursor, "users")
        for col, sql in [
            ("email", "ALTER TABLE users ADD COLUMN email TEXT"),
            ("password_hash", "ALTER TABLE users ADD COLUMN password_hash TEXT"),
            ("is_active", "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"),
            ("last_login", "ALTER TABLE users ADD COLUMN last_login TIMESTAMP"),
            ("security_question", "ALTER TABLE users ADD COLUMN security_question TEXT"),
            ("security_answer_hash", "ALTER TABLE users ADD COLUMN security_answer_hash TEXT"),
            ("recovery_code_hash", "ALTER TABLE users ADD COLUMN recovery_code_hash TEXT"),
            ("role", "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"),
        ]:
            if col not in columns:
                try:
                    cursor.execute(sql)
                    conn.commit()
                except:
                    pass
    else:
        cursor.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                avatar_url TEXT,
                email TEXT UNIQUE,
                password_hash TEXT,
                is_active INTEGER DEFAULT 1,
                last_login TIMESTAMP,
                security_question TEXT,
                security_answer_hash TEXT,
                recovery_code_hash TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    
    # Migrate messages table BEFORE creating (for existing databases)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    if cursor.fetchone():
        msg_columns = get_table_columns(cursor, "messages")
        if "file_attachment" not in msg_columns:
            try:
                cursor.execute("ALTER TABLE messages ADD COLUMN file_attachment TEXT")
                conn.commit()
            except:
                pass
        if "image_url" not in msg_columns:
            try:
                cursor.execute("ALTER TABLE messages ADD COLUMN image_url TEXT")
                conn.commit()
            except:
                pass
    
    # Create other tables
    for table_sql in [
        """CREATE TABLE IF NOT EXISTS subscriptions (
            subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan_type TEXT NOT NULL,
            price REAL NOT NULL,
            start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS api_keys (
            key_id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_name TEXT NOT NULL,
            key_value TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            file_attachment TEXT,
            image_url TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )""",
        """CREATE TABLE IF NOT EXISTS settings (
            user_id INTEGER PRIMARY KEY,
            system_prompt TEXT,
            temperature REAL DEFAULT 0.7,
            max_tokens INTEGER DEFAULT 2000,
            tone TEXT DEFAULT 'balanced',
            verbosity TEXT DEFAULT 'normal',
            stream_response INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS uploaded_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )""",
        """CREATE TABLE IF NOT EXISTS generated_images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER,
            prompt TEXT NOT NULL,
            image_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )"""
    ]:
        cursor.execute(table_sql)
    
    # Default API key
    cursor.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = 1")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO api_keys (key_name, key_value, is_active) VALUES (?, ?, 1)",
                      ("OpenRouter Default", DEFAULT_API_KEY))
    
    conn.commit()
    conn.close()
    ensure_admin_account()

def ensure_admin_account():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_id, password_hash, role FROM users WHERE email = ?", (ADMIN_EMAIL,))
    admin_row = cursor.fetchone()
    
    if admin_row:
        admin_id = admin_row[0]
        if admin_row[2] != 'admin':
            cursor.execute("UPDATE users SET role = 'admin' WHERE user_id = ?", (admin_id,))
            conn.commit()
    else:
        admin_password_hash = hash_data(ADMIN_PASSWORD)
        cursor.execute("""
            INSERT INTO users (username, email, display_name, avatar_url, password_hash, role, is_active)
            VALUES (?, ?, ?, ?, ?, 'admin', 1)
        """, ("admin", ADMIN_EMAIL, "System Administrator", 
              "https://api.dicebear.com/7.x/avataaars/svg?seed=admin", admin_password_hash))
        admin_id = cursor.lastrowid
        conn.commit()
    
    cursor.execute("SELECT user_id FROM settings WHERE user_id = ?", (admin_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO settings (user_id, system_prompt) VALUES (?, ?)",
                      (admin_id, "You are a helpful AI assistant."))
        conn.commit()
    
    cursor.execute("SELECT subscription_id FROM subscriptions WHERE user_id = ? AND is_active = 1", (admin_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO subscriptions (user_id, plan_type, price, is_active) VALUES (?, 'plus', 0, 1)",
                      (admin_id,))
        conn.commit()
    
    conn.close()

# ========================================
# AUTH FUNCTIONS
# ========================================

def hash_password(password: str) -> str:
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
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
    except:
        return False

def create_authenticated_user(username: str, email: str, display_name: str, password: str, 
                              security_question: str, security_answer: str, avatar_seed: str = None) -> tuple:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            st.error("❌ Username already exists")
            return -1, None
        
        cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            st.error("❌ Email already exists")
            return -1, None
        
        if not avatar_seed:
            avatar_seed = username
        
        avatar_url = f"https://api.dicebear.com/7.x/avataaars/svg?seed={avatar_seed}"
        password_hash = hash_password(password)
        security_answer_hash = hash_data(security_answer.lower().strip())
        recovery_code = generate_recovery_code()
        recovery_code_hash = hash_data(recovery_code)
        
        cursor.execute("""
            INSERT INTO users (username, email, display_name, avatar_url, password_hash, 
                              security_question, security_answer_hash, recovery_code_hash, role, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'user', 1)
        """, (username, email, display_name, avatar_url, password_hash, 
              security_question, security_answer_hash, recovery_code_hash))
        
        user_id = cursor.lastrowid
        
        cursor.execute("INSERT INTO settings (user_id, system_prompt) VALUES (?, ?)",
                      (user_id, "You are a helpful AI assistant."))
        cursor.execute("INSERT INTO subscriptions (user_id, plan_type, price, is_active) VALUES (?, 'free', 0, 1)",
                      (user_id,))
        
        conn.commit()
        conn.close()
        return user_id, recovery_code
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"Error: {str(e)}")
        return -2, None

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, password_hash, role
            FROM users WHERE email = ? AND is_active = 1
        """, (email,))
        
        row = cursor.fetchone()
        
        if row and row[5]:
            if verify_password(password, row[5]):
                cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (row[0],))
                conn.commit()
                
                user = {
                    "user_id": row[0],
                    "username": row[1],
                    "email": row[2],
                    "display_name": row[3],
                    "avatar_url": row[4],
                    "role": row[6] or "user"
                }
                conn.close()
                return user
        
        conn.close()
        return None
    except:
        return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, role
            FROM users WHERE user_id = ? AND is_active = 1
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "user_id": row[0], "username": row[1], "email": row[2],
                "display_name": row[3], "avatar_url": row[4], "role": row[5] or "user"
            }
        return None
    except:
        return None

def get_user_security_info(email: str) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, username, email, security_question, security_answer_hash, recovery_code_hash
            FROM users WHERE email = ? AND is_active = 1
        """, (email,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "user_id": row[0], "username": row[1], "email": row[2],
                "security_question": row[3], "security_answer_hash": row[4], "recovery_code_hash": row[5]
            }
        return None
    except:
        return None

def reset_password_with_security(email: str, security_answer: str, new_password: str) -> bool:
    try:
        user_info = get_user_security_info(email)
        if not user_info or not user_info.get("security_answer_hash"):
            return False
        if verify_hashed_data(security_answer.lower().strip(), user_info["security_answer_hash"]):
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                          (hash_password(new_password), user_info["user_id"]))
            conn.commit()
            conn.close()
            return True
        return False
    except:
        return False

def reset_password_with_recovery_code(email: str, recovery_code: str, new_password: str) -> bool:
    try:
        user_info = get_user_security_info(email)
        if not user_info or not user_info.get("recovery_code_hash"):
            return False
        if verify_hashed_data(recovery_code.strip(), user_info["recovery_code_hash"]):
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                          (hash_password(new_password), user_info["user_id"]))
            conn.commit()
            conn.close()
            return True
        return False
    except:
        return False

def update_user_password(user_id: int, new_password: str):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                  (hash_password(new_password), user_id))
    conn.commit()
    conn.close()

# ========================================
# SUBSCRIPTION FUNCTIONS
# ========================================

def get_active_api_key() -> str:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT key_value FROM api_keys WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else DEFAULT_API_KEY
    except:
        return DEFAULT_API_KEY

def create_subscription(user_id: int, plan_type: str):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE subscriptions SET is_active = 0 WHERE user_id = ? AND is_active = 1", (user_id,))
    plan = SUBSCRIPTION_PLANS[plan_type]
    cursor.execute("INSERT INTO subscriptions (user_id, plan_type, price, is_active) VALUES (?, ?, ?, 1)",
                  (user_id, plan_type, plan["price"]))
    conn.commit()
    conn.close()

def get_user_subscription(user_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subscription_id, plan_type, price, start_date
        FROM subscriptions WHERE user_id = ? AND is_active = 1
        ORDER BY start_date DESC LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"subscription_id": row[0], "plan_type": row[1], "price": row[2], "start_date": row[3]}
    return None

def get_all_subscriptions() -> List[Dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.subscription_id, u.username, u.email, s.plan_type, s.price, s.start_date
        FROM subscriptions s
        JOIN users u ON s.user_id = u.user_id
        WHERE s.is_active = 1 ORDER BY s.start_date DESC
    """)
    subs = [{"subscription_id": r[0], "username": r[1], "email": r[2], 
             "plan_type": r[3], "price": r[4], "start_date": r[5]} for r in cursor.fetchall()]
    conn.close()
    return subs

# ========================================
# ADMIN FUNCTIONS
# ========================================

def get_all_users_admin() -> List[Dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.user_id, u.username, u.email, u.display_name, u.role, u.is_active, 
               u.last_login, u.created_at, s.plan_type
        FROM users u
        LEFT JOIN subscriptions s ON u.user_id = s.user_id AND s.is_active = 1
        ORDER BY u.created_at DESC
    """)
    users = [{
        "user_id": r[0], "username": r[1], "email": r[2], "display_name": r[3],
        "role": r[4], "is_active": bool(r[5]), "last_login": r[6],
        "created_at": r[7], "subscription": r[8] or "free"
    } for r in cursor.fetchall()]
    conn.close()
    return users

def update_api_key_admin(key_name: str, key_value: str):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE api_keys SET is_active = 0")
    cursor.execute("INSERT INTO api_keys (key_name, key_value, is_active) VALUES (?, ?, 1)",
                  (key_name, key_value))
    conn.commit()
    conn.close()

def toggle_user_status_admin(user_id: int, is_active: bool):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_active = ? WHERE user_id = ?", (1 if is_active else 0, user_id))
    conn.commit()
    conn.close()

def update_user_subscription_admin(user_id: int, plan_type: str):
    create_subscription(user_id, plan_type)

# ========================================
# CONVERSATION FUNCTIONS
# ========================================

def get_user_conversations(user_id: int) -> List[Dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_id, title, model, created_at, updated_at
        FROM conversations WHERE user_id = ? ORDER BY updated_at DESC
    """, (user_id,))
    convs = [{"conversation_id": r[0], "title": r[1], "model": r[2], 
              "created_at": r[3], "updated_at": r[4]} for r in cursor.fetchall()]
    conn.close()
    return convs

def group_conversations_by_date(conversations: List[Dict]) -> Dict[str, List[Dict]]:
    now = datetime.now()
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()
    
    groups = {"Today": [], "Yesterday": []}
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
                date_str = conv_datetime.strftime("%B %d, %Y")
                if date_str not in older_dates:
                    older_dates[date_str] = []
                older_dates[date_str].append(conv)
        except:
            if "Older" not in older_dates:
                older_dates["Older"] = []
            older_dates["Older"].append(conv)
    
    for date_str in sorted(older_dates.keys(), reverse=True):
        groups[date_str] = older_dates[date_str]
    
    return groups

def create_conversation(user_id: int, title: str, model: str) -> int:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_id, title, model) VALUES (?, ?, ?)",
                  (user_id, title, model))
    conv_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def delete_conversation(conversation_id: int):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM uploaded_files WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: int) -> List[Dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, file_attachment, image_url, timestamp
        FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC
    """, (conversation_id,))
    msgs = [{"role": r[0], "content": r[1], "file_attachment": r[2], 
             "image_url": r[3], "timestamp": r[4]} for r in cursor.fetchall()]
    conn.close()
    return msgs

def add_message(conversation_id: int, role: str, content: str, file_attachment: str = None, image_url: str = None):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content, file_attachment, image_url)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, role, content, file_attachment, image_url))
    cursor.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                  (conversation_id,))
    conn.commit()
    conn.close()

def save_uploaded_file(conversation_id: int, filename: str, file_type: str, file_size: int):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO uploaded_files (conversation_id, filename, file_type, file_size)
        VALUES (?, ?, ?, ?)
    """, (conversation_id, filename, file_type, file_size))
    conn.commit()
    conn.close()

def save_generated_image(user_id: int, conversation_id: int, prompt: str, image_url: str):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO generated_images (user_id, conversation_id, prompt, image_url)
        VALUES (?, ?, ?, ?)
    """, (user_id, conversation_id, prompt, image_url))
    conn.commit()
    conn.close()

def get_user_settings(user_id: int) -> Dict:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT system_prompt, temperature, max_tokens, tone, verbosity, stream_response FROM settings WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "system_prompt": row[0], "temperature": row[1], "max_tokens": row[2],
            "tone": row[3], "verbosity": row[4], "stream_response": bool(row[5])
        }
    return {
        "system_prompt": "You are a helpful AI assistant.",
        "temperature": 0.7, "max_tokens": 2000, "tone": "balanced",
        "verbosity": "normal", "stream_response": True
    }

def update_user_settings(user_id: int, settings: Dict):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM settings WHERE user_id = ?", (user_id,))
    if cursor.fetchone():
        cursor.execute("""
            UPDATE settings SET system_prompt = ?, temperature = ?, max_tokens = ?,
            tone = ?, verbosity = ?, stream_response = ? WHERE user_id = ?
        """, (settings["system_prompt"], settings["temperature"], settings["max_tokens"],
              settings["tone"], settings["verbosity"], int(settings["stream_response"]), user_id))
    else:
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt, temperature, max_tokens, tone, verbosity, stream_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, settings["system_prompt"], settings["temperature"], settings["max_tokens"],
              settings["tone"], settings["verbosity"], int(settings["stream_response"])))
    conn.commit()
    conn.close()

# ========================================
# AI API
# ========================================

def call_ai_model(messages: List[Dict], model: str, settings: Dict, stream: bool = True):
    api_key = get_active_api_key()
    
    if api_key == "sk-or-v1-YOUR_API_KEY_HERE":
        st.error("🔑 **API Key Not Configured**")
        st.warning("Admin: Configure API key in Admin Dashboard")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sagoma-ai.streamlit.app",
        "X-Title": "Sagoma AI"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": settings.get("temperature", 0.7),
        "max_tokens": settings.get("max_tokens", 2000),
        "stream": stream
    }
    
    try:
        response = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, stream=stream, timeout=300)
        response.raise_for_status()
        
        if stream:
            return response
        else:
            return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("🔑 **Invalid API Key**")
        elif e.response.status_code == 402:
            st.error("💳 **Insufficient Credits**")
        elif e.response.status_code == 429:
            st.error("⏱️ **Rate Limited**")
        else:
            st.error(f"API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def stream_ai_response(messages: List[Dict], model: str, settings: Dict):
    response = call_ai_model(messages, model, settings, stream=True)
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
                except:
                    continue

# ========================================
# UI - AUTH
# ========================================

def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🤖 Sagoma AI")
        st.subheader("Sign In")
        
        email = st.text_input("Email", placeholder="your@email.com", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("🔑 Forgot Password?"):
            st.session_state.show_password_recovery = True
            st.rerun()
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔐 Sign In", type="primary", use_container_width=True):
                if email and password:
                    user = authenticate_user(email, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user["user_id"]
                        st.session_state.active_user = user
                        if user.get("role") == "admin":
                            st.session_state.is_admin = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                else:
                    st.error("Enter email and password")
        with col_b:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()

def render_password_recovery():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔑 Password Recovery")
        tabs = st.tabs(["Security Question", "Recovery Code"])
        
        with tabs[0]:
            email = st.text_input("Email", key="recovery_email_sq")
            if st.button("Get Security Question"):
                user_info = get_user_security_info(email)
                if user_info and user_info.get("security_question"):
                    st.session_state.recovery_user_info = user_info
                    st.rerun()
                else:
                    st.error("Email not found")
            
            if st.session_state.get('recovery_user_info'):
                st.info(f"**Q:** {st.session_state.recovery_user_info['security_question']}")
                answer = st.text_input("Your Answer", key="security_answer_input")
                new_pwd = st.text_input("New Password", type="password", key="new_pwd_sq")
                confirm_pwd = st.text_input("Confirm", type="password", key="confirm_pwd_sq")
                
                if st.button("Reset Password", type="primary"):
                    if all([answer, new_pwd, confirm_pwd]) and new_pwd == confirm_pwd and len(new_pwd) >= 6:
                        if reset_password_with_security(email, answer, new_pwd):
                            st.success("✅ Password reset!")
                            st.balloons()
                            st.session_state.recovery_user_info = None
                            st.session_state.show_password_recovery = False
                            st.rerun()
                        else:
                            st.error("Incorrect answer")
                    else:
                        st.error("Check all fields")
        
        with tabs[1]:
            email_rc = st.text_input("Email", key="recovery_email_rc")
            recovery_code = st.text_input("Recovery Code", key="recovery_code_input")
            new_pwd_rc = st.text_input("New Password", type="password", key="new_pwd_rc")
            confirm_pwd_rc = st.text_input("Confirm", type="password", key="confirm_pwd_rc")
            
            if st.button("Reset with Code", type="primary"):
                if all([email_rc, recovery_code, new_pwd_rc, confirm_pwd_rc]) and new_pwd_rc == confirm_pwd_rc:
                    if reset_password_with_recovery_code(email_rc, recovery_code, new_pwd_rc):
                        st.success("✅ Password reset!")
                        st.balloons()
                        st.session_state.show_password_recovery = False
                        st.rerun()
                    else:
                        st.error("Invalid code")
        
        if st.button("← Back to Login"):
            st.session_state.show_password_recovery = False
            st.session_state.recovery_user_info = None
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
        
        st.divider()
        st.subheader("🔒 Account Recovery")
        
        security_question = st.selectbox("Security Question", options=SECURITY_QUESTIONS, key="security_question_select")
        security_answer = st.text_input("Answer", placeholder="Your answer", key="security_answer_input")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Create Account", type="primary", use_container_width=True):
                if not all([username, email, display_name, password, password_confirm, security_answer]):
                    st.error("Fill all fields")
                elif password != password_confirm:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password too short")
                elif "@" not in email:
                    st.error("Invalid email")
                else:
                    with st.spinner("Creating account..."):
                        user_id, recovery_code = create_authenticated_user(
                            username, email, display_name, password, security_question, security_answer
                        )
                        if user_id > 0:
                            st.success("✅ Account created!")
                            if recovery_code:
                                st.warning("⚠️ **SAVE THIS CODE**")
                                st.code(recovery_code)
                                st.error("Write this down!")
                                if st.button("I've Saved It", type="primary"):
                                    st.balloons()
                                    st.session_state.show_signup = False
                                    st.rerun()
        with col_b:
            if st.button("← Back", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()

# ========================================
# UI - SUBSCRIPTIONS
# ========================================

def render_subscription_page():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("💎 Choose Your Plan")
    with col2:
        if st.button("✖️ Close", key="close_sub_btn"):
            st.session_state.show_subscriptions = False
            st.rerun()
    
    st.divider()
    st.info("🚧 Premium features coming soon!")
    st.divider()
    
    current_sub = get_user_subscription(st.session_state.active_user["user_id"])
    current_plan = current_sub["plan_type"] if current_sub else "free"
    
    cols = st.columns(3)
    for idx, (plan_key, plan_info) in enumerate(SUBSCRIPTION_PLANS.items()):
        with cols[idx]:
            st.markdown(f"### {plan_info['name']}")
            if plan_key == "pro":
                st.markdown("**🌟 RECOMMENDED**")
            if current_plan == plan_key:
                st.success("✓ Current")
            
            st.markdown(f"**${plan_info['price']}/mo**" if plan_info['price'] > 0 else "**Free**")
            st.markdown("**Features:**")
            for feature in plan_info['features']:
                st.markdown(f"✓ {feature}")
            
            if current_plan == plan_key:
                st.button("Current Plan", key=f"btn_{plan_key}", disabled=True, use_container_width=True)
            else:
                st.button(f"Select {plan_info['name']}", key=f"btn_{plan_key}", disabled=True, use_container_width=True)
                st.caption("Coming Soon")

# ========================================
# UI - ADMIN
# ========================================

def render_admin_dashboard():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🔧 Admin Dashboard")
    with col2:
        if st.button("← Back", key="admin_back_btn", type="primary"):
            st.session_state.show_admin = False
            st.rerun()
    
    st.divider()
    tabs = st.tabs(["📊 Overview", "🔑 API Keys", "👥 Users", "💎 Subscriptions"])
    
    with tabs[0]:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1 AND role = 'user'")
        total_users = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_convs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM generated_images")
        total_imgs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM uploaded_files")
        total_files = cursor.fetchone()[0]
        conn.close()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Users", total_users)
            st.metric("Conversations", total_convs)
        with col2:
            st.metric("Images Generated", total_imgs)
            st.metric("Files Uploaded", total_files)
    
    with tabs[1]:
        st.subheader("🔑 API Key Management")
        current_key = get_active_api_key()
        
        if current_key == "sk-or-v1-YOUR_API_KEY_HERE":
            st.error("⚠️ No valid API key")
        else:
            st.success("✅ API Key Active")
            st.code(current_key[:20] + "..." + current_key[-10:])
        
        st.divider()
        st.info("Get key from: https://openrouter.ai/keys")
        
        new_name = st.text_input("Key Name", value="OpenRouter", key="new_key_name")
        new_key = st.text_input("API Key", type="password", placeholder="sk-or-v1-...", key="new_key_value")
        
        if st.button("💾 Update", type="primary"):
            if new_key and new_key.startswith("sk-or-"):
                update_api_key_admin(new_name, new_key)
                st.success("✅ Updated!")
                st.balloons()
                st.rerun()
            else:
                st.error("Invalid key")
    
    with tabs[2]:
        st.subheader("👥 User Management")
        users = get_all_users_admin()
        
        for user in users:
            if user['role'] != 'admin':
                with st.expander(f"👤 {user['display_name']}"):
                    st.write(f"**Email:** {user.get('email')}")
                    st.write(f"**Plan:** {user['subscription'].upper()}")
                    st.write(f"**Status:** {'Active' if user['is_active'] else 'Inactive'}")
                    
                    if user['is_active']:
                        if st.button("🚫 Deactivate", key=f"deact_{user['user_id']}"):
                            toggle_user_status_admin(user['user_id'], False)
                            st.rerun()
                    else:
                        if st.button("✅ Activate", key=f"act_{user['user_id']}"):
                            toggle_user_status_admin(user['user_id'], True)
                            st.rerun()
    
    with tabs[3]:
        st.subheader("💎 Subscriptions")
        all_subs = get_all_subscriptions()
        
        free = len([s for s in all_subs if s['plan_type'] == 'free'])
        pro = len([s for s in all_subs if s['plan_type'] == 'pro'])
        plus = len([s for s in all_subs if s['plan_type'] == 'plus'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Free", free)
        with col2:
            st.metric("Pro", pro)
        with col3:
            st.metric("Plus", plus)
        
        st.divider()
        users = get_all_users_admin()
        user_opts = {f"{u['username']}": u['user_id'] for u in users if u['role'] != 'admin'}
        
        if user_opts:
            sel_user = st.selectbox("User", list(user_opts.keys()))
            sel_plan = st.selectbox("Plan", ["free", "pro", "plus"])
            if st.button("🔄 Update", type="primary"):
                update_user_subscription_admin(user_opts[sel_user], sel_plan)
                st.success("Updated!")
                st.rerun()

# ========================================
# UI - MAIN
# ========================================

def render_header():
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🤖 Sagoma AI")
        if st.session_state.get('is_admin'):
            st.markdown('<span class="admin-badge">ADMIN</span>', unsafe_allow_html=True)
    
    with col2:
        theme = st.session_state.get('theme', 'light')
        icon = "🌙" if theme == 'light' else "☀️"
        if st.button(f"{icon}", key="theme_toggle"):
            st.session_state.theme = 'dark' if theme == 'light' else 'light'
            st.rerun()
    
    with col3:
        if st.session_state.authenticated and st.session_state.active_user:
            user = st.session_state.active_user
            with st.popover("👤 " + user['display_name'][:10]):
                st.markdown(f"### {user['display_name']}")
                st.markdown(f"*{user.get('email')}*")
                
                sub = get_user_subscription(user["user_id"])
                if sub:
                    st.markdown(f"**Plan:** {SUBSCRIPTION_PLANS[sub['plan_type']]['name']}")
                
                st.divider()
                
                if st.session_state.get('is_admin'):
                    if st.button("🔧 Admin Dashboard", key="admin_btn", use_container_width=True):
                        st.session_state.show_admin = True
                        st.session_state.show_profile = False
                        st.session_state.show_settings = False
                        st.session_state.show_subscriptions = False
                        st.rerun()
                    st.divider()
                
                if st.button("👤 Profile", key="profile_btn", use_container_width=True):
                    st.session_state.show_profile = True
                    st.session_state.show_admin = False
                    st.session_state.show_settings = False
                    st.session_state.show_subscriptions = False
                    st.rerun()
                
                if st.button("💎 Subscriptions", key="subs_btn", use_container_width=True):
                    st.session_state.show_subscriptions = True
                    st.session_state.show_admin = False
                    st.session_state.show_profile = False
                    st.session_state.show_settings = False
                    st.rerun()
                
                if st.button("⚙️ Settings", key="settings_btn", use_container_width=True):
                    st.session_state.show_settings = True
                    st.session_state.show_admin = False
                    st.session_state.show_profile = False
                    st.session_state.show_subscriptions = False
                    st.rerun()
                
                st.divider()
                
                if st.button("🚪 Logout", key="logout_btn", type="primary", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

def render_profile():
    if st.session_state.show_profile:
        user = st.session_state.active_user
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("👤 My Profile")
        with col2:
            if st.button("✖️", key="close_profile"):
                st.session_state.show_profile = False
                st.rerun()
        
        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.image(user["avatar_url"], width=150)
        with col_b:
            st.markdown(f"### {user['display_name']}")
            st.markdown(f"**Username:** {user['username']}")
            st.markdown(f"**Email:** {user.get('email')}")

def render_sidebar():
    with st.sidebar:
        st.title("💬 Chat History")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.session_state.uploaded_file_data = None
            st.rerun()
        
        st.divider()
        search = st.text_input("🔍 Search", key="search_conv", label_visibility="collapsed")
        
        if st.session_state.authenticated:
            convs = get_user_conversations(st.session_state.active_user["user_id"])
            if search:
                convs = [c for c in convs if search.lower() in c["title"].lower()]
            
            if convs:
                grouped = group_conversations_by_date(convs)
                for group_name, group_convs in grouped.items():
                    if group_convs:
                        st.markdown(f'<p class="conversation-group-header">{group_name}</p>', unsafe_allow_html=True)
                        for conv in group_convs:
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                title = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                                if st.button(f"💬 {title}", key=f"conv_{conv['conversation_id']}", use_container_width=True):
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
                st.info("No conversations yet!")

def render_settings():
    if st.session_state.show_settings:
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["Chat Behavior", "Security"])
        
        with tabs[0]:
            settings = get_user_settings(st.session_state.active_user["user_id"])
            
            new_prompt = st.text_area("System Prompt", value=settings.get("system_prompt", "You are a helpful AI assistant."), height=150)
            
            col1, col2 = st.columns(2)
            with col1:
                new_temp = st.slider("Temperature", 0.0, 2.0, settings.get("temperature", 0.7), 0.1)
                new_tone = st.selectbox("Tone", ["professional", "casual", "friendly", "technical", "balanced"],
                                       index=["professional", "casual", "friendly", "technical", "balanced"].index(settings.get("tone", "balanced")))
            with col2:
                new_tokens = st.number_input("Max Tokens", 100, 4000, settings.get("max_tokens", 2000), 100)
                new_verbosity = st.selectbox("Verbosity", ["concise", "normal", "detailed"],
                                            index=["concise", "normal", "detailed"].index(settings.get("verbosity", "normal")))
            
            new_stream = st.checkbox("Stream responses", value=settings.get("stream_response", True))
            
            if st.button("💾 Save", type="primary"):
                update_user_settings(st.session_state.active_user["user_id"], {
                    "system_prompt": new_prompt, "temperature": new_temp, "max_tokens": new_tokens,
                    "tone": new_tone, "verbosity": new_verbosity, "stream_response": new_stream
                })
                st.success("Saved!")
                st.rerun()
        
        with tabs[1]:
            st.subheader("Change Password")
            curr_pwd = st.text_input("Current Password", type="password", key="curr_pwd")
            new_pwd = st.text_input("New Password", type="password", key="new_pwd")
            confirm_pwd = st.text_input("Confirm", type="password", key="confirm_pwd")
            
            if st.button("🔐 Update", type="primary"):
                if all([curr_pwd, new_pwd, confirm_pwd]) and new_pwd == confirm_pwd and len(new_pwd) >= 6:
                    if authenticate_user(st.session_state.active_user['email'], curr_pwd):
                        update_user_password(st.session_state.active_user["user_id"], new_pwd)
                        st.success("Updated!")
                    else:
                        st.error("Current password incorrect")
                else:
                    st.error("Check all fields")
        
        st.divider()
        if st.button("✖️ Close"):
            st.session_state.show_settings = False
            st.rerun()

def render_chat():
    if st.session_state.get('show_admin'):
        render_admin_dashboard()
        return
    if st.session_state.show_settings:
        render_settings()
        return
    if st.session_state.get('show_profile'):
        render_profile()
        return
    if st.session_state.get('show_subscriptions'):
        render_subscription_page()
        return
    
    # Feature tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📎 Files", "🎨 Images", "🎙️ Voice"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            models = list(AVAILABLE_MODELS.keys())
            idx = list(AVAILABLE_MODELS.values()).index(st.session_state.selected_model) if st.session_state.selected_model in AVAILABLE_MODELS.values() else 0
            sel = st.selectbox("Model", models, index=idx, key="model_sel")
            st.session_state.selected_model = AVAILABLE_MODELS[sel]
        with col2:
            st.info("**Active**")
        
        st.divider()
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("file_attachment"):
                    st.caption(f"📎 {msg['file_attachment']}")
                if msg.get("image_url"):
                    st.image(msg["image_url"], width=400)
        
        if prompt := st.chat_input("Type your message..."):
            settings = get_user_settings(st.session_state.active_user["user_id"])
            
            if not st.session_state.current_conversation:
                title = prompt[:50]
                conv_id = create_conversation(st.session_state.active_user["user_id"], title, st.session_state.selected_model)
                st.session_state.current_conversation = {
                    "conversation_id": conv_id, "title": title, "model": st.session_state.selected_model
                }
            
            file_context = ""
            file_name = None
            if st.session_state.get('uploaded_file_data'):
                f = st.session_state.uploaded_file_data
                if 'text' in f:
                    file_context = f"\n\n[File: {f['name']}]\n{f['text']}"
                file_name = f['name']
                save_uploaded_file(st.session_state.current_conversation["conversation_id"], f['name'], f['type'], f['size'])
            
            st.session_state.messages.append({"role": "user", "content": prompt + file_context})
            add_message(st.session_state.current_conversation["conversation_id"], "user", prompt, file_attachment=file_name)
            
            with st.chat_message("user"):
                st.write(prompt)
                if file_name:
                    st.caption(f"📎 {file_name}")
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                api_msgs = [{"role": "system", "content": settings.get("system_prompt", "You are a helpful AI assistant.")}]
                api_msgs.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
                
                with st.spinner("🤖 Processing..."):
                    if settings.get("stream_response", True):
                        full = ""
                        for chunk in stream_ai_response(api_msgs, st.session_state.selected_model, settings):
                            full += chunk
                            placeholder.markdown(full + "▌")
                        placeholder.markdown(full)
                    else:
                        full = call_ai_model(api_msgs, st.session_state.selected_model, settings, stream=False)
                        if full:
                            placeholder.markdown(full)
                
                if full:
                    st.session_state.messages.append({"role": "assistant", "content": full})
                    add_message(st.session_state.current_conversation["conversation_id"], "assistant", full)
                    st.session_state.uploaded_file_data = None
                    st.rerun()
    
    with tab2:
        st.subheader("📎 File Upload")
        st.markdown('<span class="feature-badge">NEW</span>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Choose file", type=["pdf", "txt", "md", "csv", "png", "jpg"], key="file_upload")
        
        if uploaded:
            st.success(f"✅ {uploaded.name} ({uploaded.size / 1024:.2f} KB)")
            
            file_bytes = uploaded.read()
            file_data = {"name": uploaded.name, "type": uploaded.type, "size": uploaded.size}
            
            if uploaded.type == "application/pdf":
                text = extract_text_from_pdf(file_bytes)
                file_data["text"] = text
                st.text_area("Extracted", text[:500] + "...", height=200)
            elif uploaded.type.startswith("text/"):
                text = file_bytes.decode('utf-8')
                file_data["text"] = text
                st.text_area("Content", text[:500] + "...", height=200)
            elif uploaded.type.startswith("image/"):
                img = process_image_file(file_bytes)
                if img:
                    st.image(img, width=400)
                    file_data["image_data"] = img
            
            if st.button("💬 Chat about this file", type="primary"):
                st.session_state.uploaded_file_data = file_data
                st.success("File attached! Go to Chat tab.")
    
    with tab3:
        st.subheader("🎨 Image Generation")
        st.markdown('<span class="feature-badge">NEW</span>', unsafe_allow_html=True)
        
        img_prompt = st.text_area("Describe your image:", placeholder="A beautiful sunset over mountains...", height=100, key="img_prompt")
        
        if st.button("🎨 Generate", type="primary"):
            if img_prompt:
                with st.spinner("Generating... (30-60s)"):
                    img_url = generate_image(img_prompt)
                    if img_url:
                        st.success("✅ Generated!")
                        st.image(img_url, width=600)
                        
                        if st.session_state.current_conversation:
                            save_generated_image(
                                st.session_state.active_user["user_id"],
                                st.session_state.current_conversation["conversation_id"],
                                img_prompt, img_url
                            )
                        
                        if st.button("💬 Add to chat"):
                            if st.session_state.current_conversation:
                                add_message(st.session_state.current_conversation["conversation_id"],
                                          "assistant", f"Generated: {img_prompt}", image_url=img_url)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": f"Generated: {img_prompt}", "image_url": img_url
                                })
                                st.success("Added!")
                                st.rerun()
                    else:
                        st.error("Failed to generate")
            else:
                st.error("Enter description")
    
    with tab4:
        st.subheader("🎙️ Voice Chat")
        st.markdown('<span class="feature-badge">COMING SOON</span>', unsafe_allow_html=True)
        st.info("🚧 Voice features under development!")
        st.markdown("""
        **Planned Features:**
        - 🎤 Voice input (speech-to-text)
        - 🔊 Voice output (text-to-speech)
        - 🗣️ Hands-free conversation mode
        - 🎭 Multiple voice options
        - 🌍 Multi-language support
        """)

# ========================================
# MAIN
# ========================================

def main():
    st.set_page_config(page_title="Sagoma AI", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    init_database()
    
    # Init session state
    for key, default in [
        ("authenticated", False), ("is_admin", False), ("show_admin", False),
        ("user_id", None), ("active_user", None), ("show_signup", False),
        ("show_password_recovery", False), ("recovery_user_info", None),
        ("current_conversation", None), ("messages", []), ("selected_model", DEFAULT_MODEL),
        ("show_settings", False), ("show_profile", False), ("show_subscriptions", False),
        ("theme", "light"), ("uploaded_file_data", None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    
    # Auth flow
    if not st.session_state.authenticated:
        if st.session_state.show_password_recovery:
            render_password_recovery()
        elif st.session_state.show_signup:
            render_signup()
        else:
            render_login()
    else:
        if not st.session_state.active_user and st.session_state.user_id:
            st.session_state.active_user = get_user_by_id(st.session_state.user_id)
        
        render_header()
        render_sidebar()
        render_chat()

if __name__ == "__main__":
    main()
