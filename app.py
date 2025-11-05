"""
Multi-Model AI Chatbot with Subscriptions and Admin Dashboard
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

# Default API Key (Admin can update)
DEFAULT_API_KEY = "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"

# Subscription Plans
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price": 0,
        "features": [
            "Access to basic models",
            "Limited conversation history",
            "Standard response time",
            "Community support"
        ]
    },
    "pro": {
        "name": "Pro",
        "price": 10,
        "features": [
            "Access to all premium models",
            "Unlimited conversation history",
            "Priority response time",
            "Advanced customization",
            "Email support"
        ]
    },
    "plus": {
        "name": "Plus",
        "price": 20,
        "features": [
            "Everything in Pro",
            "Exclusive early access to new models",
            "Maximum response speed",
            "Custom model fine-tuning",
            "Priority support",
            "API access"
        ]
    }
}

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

# Security Questions
SECURITY_QUESTIONS = [
    "What is your mother's maiden name?",
    "What was the name of your first pet?",
    "What city were you born in?",
    "What is your favorite book?",
    "What was your childhood nickname?",
    "What is the name of your elementary school?",
]

# Admin credentials (PERSISTENT - stored in environment or code)
# Option 1: Use environment variables (recommended for production)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@sagoma.ai")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@2024")

# Option 2: Hardcoded fallback (always works)
FALLBACK_ADMIN_EMAIL = "admin@sagoma.ai"
FALLBACK_ADMIN_PASSWORD = "Admin@2024"

# Database path
DB_PATH = Path(__file__).parent / "database" / "conversations.db"
DB_PATH.parent.mkdir(exist_ok=True)

# ========================================
# SECURITY FUNCTIONS
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
# THEME MANAGEMENT
# ========================================

def apply_theme():
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            [data-testid="stSidebar"] {
                background-color: #262730;
            }
            .conversation-group-header {
                color: #a0a0a0;
            }
            .subscription-card {
                background-color: #262730;
                border: 2px solid #464646;
            }
            .subscription-card.recommended {
                border-color: #4CAF50;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #ffffff;
                color: #31333F;
            }
            [data-testid="stSidebar"] {
                background-color: #f0f2f6;
            }
            .conversation-group-header {
                color: #6e6e6e;
            }
            .subscription-card {
                background-color: #f8f9fa;
                border: 2px solid #e0e0e0;
            }
            .subscription-card.recommended {
                border-color: #4CAF50;
                background-color: #f0fdf4;
            }
            </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .conversation-group-header {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.5rem 0;
            margin-top: 1rem;
        }
        .subscription-card {
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        .subscription-card h3 {
            margin: 0;
            font-size: 1.5rem;
        }
        .subscription-card .price {
            font-size: 2rem;
            font-weight: 700;
            margin: 1rem 0;
        }
        .subscription-card .features {
            list-style: none;
            padding: 0;
        }
        .subscription-card .features li {
            padding: 0.5rem 0;
        }
        .admin-badge {
            background-color: #ef4444;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# ========================================
# DATABASE FUNCTIONS
# ========================================

def get_table_columns(cursor, table_name: str) -> list:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [column[1] for column in cursor.fetchall()]

def init_database():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        columns = get_table_columns(cursor, "users")
        migrations = [
            ("email", "ALTER TABLE users ADD COLUMN email TEXT"),
            ("password_hash", "ALTER TABLE users ADD COLUMN password_hash TEXT"),
            ("is_active", "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"),
            ("last_login", "ALTER TABLE users ADD COLUMN last_login TIMESTAMP"),
            ("security_question", "ALTER TABLE users ADD COLUMN security_question TEXT"),
            ("security_answer_hash", "ALTER TABLE users ADD COLUMN security_answer_hash TEXT"),
            ("recovery_code_hash", "ALTER TABLE users ADD COLUMN recovery_code_hash TEXT"),
            ("role", "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"),
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
    
    # Subscriptions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan_type TEXT NOT NULL,
            price REAL NOT NULL,
            start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_date TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # API Keys table (for admin management)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_name TEXT NOT NULL,
            key_value TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    # Insert default API key if none exists
    cursor.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = 1")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO api_keys (key_name, key_value, is_active)
            VALUES (?, ?, 1)
        """, ("OpenRouter Default", DEFAULT_API_KEY))
        conn.commit()
    
    conn.close()
    
    # ALWAYS ensure admin account exists (runs every time)
    ensure_admin_account()

def ensure_admin_account():
    """Ensure admin account always exists - runs on every app start"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Try to find admin by email
    cursor.execute("SELECT user_id, password_hash, role FROM users WHERE email = ?", (ADMIN_EMAIL,))
    admin_row = cursor.fetchone()
    
    if admin_row:
        admin_id = admin_row[0]
        
        # Update role to admin if it's not (in case it was changed)
        if admin_row[2] != 'admin':
            cursor.execute("UPDATE users SET role = 'admin' WHERE user_id = ?", (admin_id,))
            conn.commit()
        
        # Verify password is correct (update if changed in config)
        current_hash = admin_row[1]
        expected_hash = hash_data(ADMIN_PASSWORD)
        
        # If password in code changed, update database
        if not verify_hashed_data(ADMIN_PASSWORD, current_hash):
            cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (expected_hash, admin_id))
            conn.commit()
    
    else:
        # Admin doesn't exist, create it
        admin_password_hash = hash_data(ADMIN_PASSWORD)
        cursor.execute("""
            INSERT INTO users (username, email, display_name, avatar_url, password_hash, role, is_active)
            VALUES (?, ?, ?, ?, ?, 'admin', 1)
        """, ("admin", ADMIN_EMAIL, "System Administrator", "https://api.dicebear.com/7.x/avataaars/svg?seed=admin", admin_password_hash))
        
        admin_id = cursor.lastrowid
        conn.commit()
    
    # Ensure admin has settings (for both existing and new admin)
    cursor.execute("SELECT user_id FROM settings WHERE user_id = ?", (admin_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt)
            VALUES (?, ?)
        """, (admin_id, "You are a helpful AI assistant."))
        conn.commit()
    
    # Ensure admin has subscription (for both existing and new admin)
    cursor.execute("SELECT subscription_id FROM subscriptions WHERE user_id = ? AND is_active = 1", (admin_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO subscriptions (user_id, plan_type, price, is_active)
            VALUES (?, 'plus', 0, 1)
        """, (admin_id,))
        conn.commit()
    
    conn.close()
# ========================================
# SUBSCRIPTION FUNCTIONS
# ========================================

def get_active_api_key() -> str:
    """Get the active API key from database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT key_value FROM api_keys WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else DEFAULT_API_KEY
    except Exception:
        return DEFAULT_API_KEY

def create_subscription(user_id: int, plan_type: str):
    """Create or update user subscription"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Deactivate old subscriptions
    cursor.execute("""
        UPDATE subscriptions SET is_active = 0 WHERE user_id = ? AND is_active = 1
    """, (user_id,))
    
    # Create new subscription
    plan = SUBSCRIPTION_PLANS[plan_type]
    cursor.execute("""
        INSERT INTO subscriptions (user_id, plan_type, price, is_active)
        VALUES (?, ?, ?, 1)
    """, (user_id, plan_type, plan["price"]))
    
    conn.commit()
    conn.close()

def get_user_subscription(user_id: int) -> Optional[Dict]:
    """Get user's active subscription"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subscription_id, plan_type, price, start_date
        FROM subscriptions
        WHERE user_id = ? AND is_active = 1
        ORDER BY start_date DESC LIMIT 1
    """, (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "subscription_id": row[0],
            "plan_type": row[1],
            "price": row[2],
            "start_date": row[3]
        }
    return None

def get_all_subscriptions() -> List[Dict]:
    """Get all active subscriptions (for admin)"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.subscription_id, u.username, u.email, s.plan_type, s.price, s.start_date
        FROM subscriptions s
        JOIN users u ON s.user_id = u.user_id
        WHERE s.is_active = 1
        ORDER BY s.start_date DESC
    """)
    
    subscriptions = [
        {
            "subscription_id": row[0],
            "username": row[1],
            "email": row[2],
            "plan_type": row[3],
            "price": row[4],
            "start_date": row[5]
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return subscriptions

# ========================================
# ADMIN FUNCTIONS
# ========================================

def get_all_users_admin() -> List[Dict]:
    """Get all users for admin dashboard"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.user_id, u.username, u.email, u.display_name, u.role, u.is_active, 
               u.last_login, u.created_at, s.plan_type
        FROM users u
        LEFT JOIN subscriptions s ON u.user_id = s.user_id AND s.is_active = 1
        ORDER BY u.created_at DESC
    """)
    
    users = [
        {
            "user_id": row[0],
            "username": row[1],
            "email": row[2],
            "display_name": row[3],
            "role": row[4],
            "is_active": bool(row[5]),
            "last_login": row[6],
            "created_at": row[7],
            "subscription": row[8] or "free"
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return users

def update_api_key_admin(key_name: str, key_value: str):
    """Update API key (admin only)"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Deactivate old keys
    cursor.execute("UPDATE api_keys SET is_active = 0")
    
    # Add new key
    cursor.execute("""
        INSERT INTO api_keys (key_name, key_value, is_active)
        VALUES (?, ?, 1)
    """, (key_name, key_value))
    
    conn.commit()
    conn.close()

def toggle_user_status_admin(user_id: int, is_active: bool):
    """Activate or deactivate user (admin only)"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users SET is_active = ? WHERE user_id = ?
    """, (1 if is_active else 0, user_id))
    conn.commit()
    conn.close()

def update_user_subscription_admin(user_id: int, plan_type: str):
    """Update user subscription (admin only)"""
    create_subscription(user_id, plan_type)

# ========================================
# AUTHENTICATION FUNCTIONS (Updated)
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

def create_authenticated_user(username: str, email: str, display_name: str, password: str, 
                              security_question: str, security_answer: str, avatar_seed: str = None) -> tuple:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            st.error("❌ Username already exists")
            return -1, None
        
        # Check if email already exists
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
        
        # Create default settings
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt)
            VALUES (?, ?)
        """, (user_id, "You are a helpful AI assistant."))
        
        # Create free subscription
        cursor.execute("""
            INSERT INTO subscriptions (user_id, plan_type, price, is_active)
            VALUES (?, 'free', 0, 1)
        """, (user_id,))
        
        conn.commit()
        conn.close()
        return user_id, recovery_code
        
    except sqlite3.IntegrityError as e:
        if conn:
            conn.close()
        error_msg = str(e).lower()
        if "username" in error_msg:
            st.error("❌ Username already exists")
        elif "email" in error_msg:
            st.error("❌ Email already exists")
        else:
            st.error(f"❌ Database error: {e}")
        return -1, None
        
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"❌ Error creating user: {str(e)}")
        return -2, None

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, password_hash, role, is_active
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
                    "role": row[6] or "user"
                }
                conn.close()
                return user
        
        conn.close()
        return None
        
    except Exception:
        if conn:
            conn.close()
        return None

def get_user_security_info(email: str) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, security_question, security_answer_hash, recovery_code_hash
            FROM users
            WHERE email = ? AND is_active = 1
        """, (email,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "security_question": row[3],
                "security_answer_hash": row[4],
                "recovery_code_hash": row[5]
            }
        return None
    except Exception:
        return None

def reset_password_with_security(email: str, security_answer: str, new_password: str) -> bool:
    try:
        user_info = get_user_security_info(email)
        
        if not user_info or not user_info.get("security_answer_hash"):
            return False
        
        if verify_hashed_data(security_answer.lower().strip(), user_info["security_answer_hash"]):
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            new_password_hash = hash_password(new_password)
            cursor.execute("""
                UPDATE users SET password_hash = ? WHERE user_id = ?
            """, (new_password_hash, user_info["user_id"]))
            
            conn.commit()
            conn.close()
            return True
        
        return False
    except Exception:
        return False

def reset_password_with_recovery_code(email: str, recovery_code: str, new_password: str) -> bool:
    try:
        user_info = get_user_security_info(email)
        
        if not user_info or not user_info.get("recovery_code_hash"):
            return False
        
        if verify_hashed_data(recovery_code.strip(), user_info["recovery_code_hash"]):
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            new_password_hash = hash_password(new_password)
            cursor.execute("""
                UPDATE users SET password_hash = ? WHERE user_id = ?
            """, (new_password_hash, user_info["user_id"]))
            
            conn.commit()
            conn.close()
            return True
        
        return False
    except Exception:
        return False

def get_user_by_id(user_id: int) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, display_name, avatar_url, role
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
                "role": row[5] or "user"
            }
        return None
        
    except Exception:
        return None

def update_user_password(user_id: int, new_password: str):
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
                date_str = conv_datetime.strftime("%B %d, %Y")
                if date_str not in older_dates:
                    older_dates[date_str] = []
                older_dates[date_str].append(conv)
        except (ValueError, TypeError):
            if "Older" not in older_dates:
                older_dates["Older"] = []
            older_dates["Older"].append(conv)
    
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
# AI API FUNCTIONS (Updated - No API Key from User)
# ========================================

def call_ai_model(messages: List[Dict], model: str, settings: Dict, stream: bool = True):
    """Call AI model using system API key"""
    api_key = get_active_api_key()
    
    # Check if API key is set
    if api_key == "sk-or-v1-YOUR_API_KEY_HERE" or not api_key:
        st.error("🔑 **API Key Not Configured**")
        st.warning("⚠️ Admin needs to configure a valid OpenRouter API key in the Admin Dashboard.")
        st.info("💡 Login as admin (admin@sagoma.ai) → Admin Dashboard → API Keys")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sagoma-ai-chatbot.streamlit.app",
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
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("🔑 **API Key Invalid or Expired**")
            st.warning("⚠️ Admin needs to update the OpenRouter API key in Admin Dashboard.")
            st.info("💡 Get a new key from: https://openrouter.ai/keys")
        elif e.response.status_code == 402:
            st.error("💳 **Payment Required**")
            st.warning("⚠️ OpenRouter account has insufficient credits.")
            st.info("💡 Admin: Add credits at https://openrouter.ai/")
        elif e.response.status_code == 429:
            st.error("⏱️ **Rate Limit Exceeded**")
            st.warning("⚠️ Too many requests. Please wait a moment and try again.")
        else:
            st.error(f"API Error: {str(e)}")
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: {str(e)}")
        return None

def stream_ai_response(messages: List[Dict], model: str, settings: Dict):
    """Stream AI response (no API key parameter needed)"""
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
                except json.JSONDecodeError:
                    continue
    
    return full_response

# ========================================
# UI FUNCTIONS - AUTHENTICATION
# ========================================

def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🤖 Sagoma AI")
        st.subheader("Sign In")
        
        email = st.text_input("Email", placeholder="your@email.com", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("🔑 Forgot Password?", key="forgot_pwd_link"):
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
                        
                        # Check if admin
                        if user.get("role") == "admin":
                            st.session_state.is_admin = True
                        
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

def render_password_recovery():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔑 Password Recovery")
        
        tabs = st.tabs(["Security Question", "Recovery Code"])
        
        with tabs[0]:
            st.subheader("Reset via Security Question")
            
            email = st.text_input("Email", placeholder="your@email.com", key="recovery_email_sq")
            
            if st.button("Get Security Question", key="get_sq_btn"):
                user_info = get_user_security_info(email)
                if user_info and user_info.get("security_question"):
                    st.session_state.recovery_user_info = user_info
                    st.success("Security question retrieved!")
                    st.rerun()
                else:
                    st.error("Email not found or no security question set")
            
            if st.session_state.get('recovery_user_info'):
                st.info(f"**Security Question:** {st.session_state.recovery_user_info['security_question']}")
                
                security_answer = st.text_input("Your Answer", key="security_answer_input")
                new_password = st.text_input("New Password", type="password", key="new_pwd_sq")
                confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pwd_sq")
                
                if st.button("Reset Password", type="primary"):
                    if not all([security_answer, new_password, confirm_password]):
                        st.error("Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        if reset_password_with_security(email, security_answer, new_password):
                            st.success("✅ Password reset successful! Please sign in.")
                            st.balloons()
                            st.session_state.recovery_user_info = None
                            st.session_state.show_password_recovery = False
                            st.rerun()
                        else:
                            st.error("Incorrect security answer")
        
        with tabs[1]:
            st.subheader("Reset via Recovery Code")
            
            st.info("💡 Your recovery code was provided when you created your account.")
            
            email_rc = st.text_input("Email", placeholder="your@email.com", key="recovery_email_rc")
            recovery_code = st.text_input("Recovery Code", placeholder="XXXXXXXXXXXXXXXX", key="recovery_code_input")
            new_password_rc = st.text_input("New Password", type="password", key="new_pwd_rc")
            confirm_password_rc = st.text_input("Confirm New Password", type="password", key="confirm_pwd_rc")
            
            if st.button("Reset Password with Code", type="primary"):
                if not all([email_rc, recovery_code, new_password_rc, confirm_password_rc]):
                    st.error("Please fill all fields")
                elif new_password_rc != confirm_password_rc:
                    st.error("Passwords do not match")
                elif len(new_password_rc) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if reset_password_with_recovery_code(email_rc, recovery_code, new_password_rc):
                        st.success("✅ Password reset successful! Please sign in.")
                        st.balloons()
                        st.session_state.show_password_recovery = False
                        st.rerun()
                    else:
                        st.error("Invalid email or recovery code")
        
        st.divider()
        
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
        st.subheader("🔒 Account Recovery Setup")
        
        security_question = st.selectbox(
            "Security Question",
            options=SECURITY_QUESTIONS,
            key="security_question_select"
        )
        
        security_answer = st.text_input(
            "Security Answer",
            placeholder="Enter your answer",
            key="security_answer_input"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("✅ Create Account", type="primary", use_container_width=True):
                if not all([username, email, display_name, password, password_confirm, security_answer]):
                    st.error("Please fill all fields including security answer")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif "@" not in email or "." not in email:
                    st.error("Please enter a valid email address")
                else:
                    with st.spinner("Creating account..."):
                        user_id, recovery_code = create_authenticated_user(
                            username, email, display_name, password, 
                            security_question, security_answer
                        )
                        if user_id > 0:
                            st.success("✅ Account created successfully!")
                            
                            if recovery_code:
                                st.warning("⚠️ **IMPORTANT: Save Your Recovery Code**")
                                st.markdown(f'<div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; text-align: center; font-family: monospace; font-size: 1.5rem; font-weight: 700; margin: 1rem 0;">{recovery_code}</div>', unsafe_allow_html=True)
                                st.error("🔴 **Write this code down! You'll need it to recover your password.**")
                                
                                if st.button("I've Saved My Recovery Code", type="primary"):
                                    st.balloons()
                                    st.session_state.show_signup = False
                                    st.rerun()
        
        with col_b:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.show_signup = False
                st.rerun()

# ========================================
# UI FUNCTIONS - SUBSCRIPTIONS
# ========================================

def render_subscription_page():
    """Render subscription plans page"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("💎 Choose Your Plan")
    
    with col2:
        if st.button("✖️ Close", key="close_sub_btn"):
            st.session_state.show_subscriptions = False
            st.rerun()
    
    st.divider()
    
    st.info("🚧 **We're working on premium features!** They will be available soon. For now, enjoy the Free plan.")
    
    st.divider()
    
    # Get user's current subscription
    current_sub = get_user_subscription(st.session_state.active_user["user_id"])
    current_plan = current_sub["plan_type"] if current_sub else "free"
    
    # Display plans
    cols = st.columns(3)
    
    for idx, (plan_key, plan_info) in enumerate(SUBSCRIPTION_PLANS.items()):
        with cols[idx]:
            is_current = (current_plan == plan_key)
            is_recommended = (plan_key == "pro")
            
            card_class = "subscription-card"
            if is_recommended:
                card_class += " recommended"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            # Plan header
            st.markdown(f"### {plan_info['name']}")
            if is_recommended:
                st.markdown("**🌟 RECOMMENDED**")
            if is_current:
                st.success("✓ Current Plan")
            
            # Price
            if plan_info['price'] == 0:
                st.markdown('<div class="price">Free</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="price">${plan_info["price"]}/month</div>', unsafe_allow_html=True)
            
            # Features
            st.markdown("**Features:**")
            for feature in plan_info['features']:
                st.markdown(f"✓ {feature}")
            
            # Action button
            if is_current:
                st.button("Current Plan", key=f"btn_{plan_key}", disabled=True, use_container_width=True)
            elif plan_key == "free":
                if st.button("Downgrade to Free", key=f"btn_{plan_key}", use_container_width=True):
                    create_subscription(st.session_state.active_user["user_id"], plan_key)
                    st.success("Downgraded to Free plan")
                    st.rerun()
            else:
                st.button(f"Upgrade to {plan_info['name']}", key=f"btn_{plan_key}", type="primary", use_container_width=True, disabled=True)
                st.caption("Coming Soon")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
# ========================================
# UI FUNCTIONS - ADMIN DASHBOARD
# ========================================

def render_admin_dashboard():
    """Render admin dashboard"""
    # Add back button at the top
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("🔧 Admin Dashboard")
    
    with col2:
        if st.button("← Back to Chat", key="admin_back_btn", type="primary"):
            st.session_state.show_admin = False
            st.rerun()
    
    if not st.session_state.get('is_admin', False):
        st.error("Access Denied: Admin privileges required")
        return
    
    st.divider()
    
    tabs = st.tabs(["📊 Overview", "🔑 API Keys", "👥 User Management", "💎 Subscriptions"])
    
    # Overview Tab
    with tabs[0]:
        st.subheader("📊 System Overview")
        
        # Get statistics
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1 AND role = 'user'")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM subscriptions WHERE is_active = 1 AND plan_type = 'pro'")
        pro_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM subscriptions WHERE is_active = 1 AND plan_type = 'plus'")
        plus_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        conn.close()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", total_users)
        
        with col2:
            st.metric("Pro Users", pro_users)
        
        with col3:
            st.metric("Plus Users", plus_users)
        
        with col4:
            st.metric("Total Conversations", total_conversations)
    
    # API Keys Tab
    with tabs[1]:
        st.subheader("🔑 API Key Management")
        
        current_key = get_active_api_key()
        
        st.success("✅ Active API Key Configured")
        st.code(current_key[:20] + "..." + current_key[-10:])
        
        st.divider()
        
        st.subheader("Update API Key")
        
        new_key_name = st.text_input("Key Name", value="OpenRouter Key", key="new_key_name")
        new_key_value = st.text_input("API Key", type="password", placeholder="sk-or-v1-...", key="new_key_value")
        
        if st.button("💾 Update API Key", type="primary"):
            if new_key_value and new_key_value.startswith("sk-or-"):
                update_api_key_admin(new_key_name, new_key_value)
                st.success("API key updated successfully!")
                st.rerun()
            else:
                st.error("Please enter a valid OpenRouter API key")
    
    # User Management Tab
    with tabs[2]:
        st.subheader("👥 User Management")
        
        users = get_all_users_admin()
        
        st.write(f"**Total Users:** {len(users)}")
        
        # Search users
        search_user = st.text_input("🔍 Search users", key="search_users")
        
        if search_user:
            users = [u for u in users if search_user.lower() in u['username'].lower() or 
                     search_user.lower() in u.get('email', '').lower()]
        
        # Display users table
        for user in users:
            if user['role'] != 'admin':  # Don't show admin accounts
                with st.expander(f"👤 {user['display_name']} ({user['username']})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Email:** {user.get('email', 'N/A')}")
                        st.write(f"**User ID:** {user['user_id']}")
                        st.write(f"**Subscription:** {user['subscription'].upper()}")
                        st.write(f"**Status:** {'Active' if user['is_active'] else 'Inactive'}")
                        st.write(f"**Created:** {user['created_at']}")
                        st.write(f"**Last Login:** {user['last_login'] or 'Never'}")
                    
                    with col2:
                        if user['is_active']:
                            if st.button("🚫 Deactivate", key=f"deactivate_{user['user_id']}"):
                                toggle_user_status_admin(user['user_id'], False)
                                st.success(f"Deactivated user: {user['username']}")
                                st.rerun()
                        else:
                            if st.button("✅ Activate", key=f"activate_{user['user_id']}"):
                                toggle_user_status_admin(user['user_id'], True)
                                st.success(f"Activated user: {user['username']}")
                                st.rerun()
    
    # Subscriptions Tab
    with tabs[3]:
        st.subheader("💎 Subscription Management")
        
        all_subs = get_all_subscriptions()
        
        # Stats
        free_count = len([s for s in all_subs if s['plan_type'] == 'free'])
        pro_count = len([s for s in all_subs if s['plan_type'] == 'pro'])
        plus_count = len([s for s in all_subs if s['plan_type'] == 'plus'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Free Plans", free_count)
        with col2:
            st.metric("Pro Plans", pro_count)
        with col3:
            st.metric("Plus Plans", plus_count)
        
        st.divider()
        
        st.subheader("Update User Subscription")
        
        users = get_all_users_admin()
        user_options = {f"{u['username']} ({u['email']})": u['user_id'] for u in users if u['role'] != 'admin'}
        
        selected_user = st.selectbox("Select User", options=list(user_options.keys()), key="admin_select_user")
        selected_plan = st.selectbox("Select Plan", options=["free", "pro", "plus"], key="admin_select_plan")
        
        if st.button("🔄 Update Subscription", type="primary"):
            user_id = user_options[selected_user]
            update_user_subscription_admin(user_id, selected_plan)
            st.success(f"Updated subscription to {selected_plan.upper()}")
            st.rerun()

# ========================================
# UI FUNCTIONS - MAIN APP
# ========================================

def render_header():
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🤖 Sagoma AI Chatbot")
        
        # Show admin badge if admin
        if st.session_state.get('is_admin', False):
            st.markdown('<span class="admin-badge">ADMIN</span>', unsafe_allow_html=True)
    
    with col2:
        current_theme = st.session_state.get('theme', 'light')
        theme_icon = "🌙" if current_theme == 'light' else "☀️"
        theme_label = "Dark" if current_theme == 'light' else "Light"
        
        if st.button(f"{theme_icon} {theme_label}", key="theme_toggle"):
            st.session_state.theme = 'dark' if current_theme == 'light' else 'light'
            st.rerun()
    
    with col3:
        if st.session_state.authenticated and st.session_state.active_user:
            user = st.session_state.active_user
            
            with st.popover("👤 " + user['display_name'][:10], use_container_width=False):
                st.markdown(f"### {user['display_name']}")
                st.markdown(f"*{user.get('email', 'N/A')}*")
                
                # Show subscription
                sub = get_user_subscription(user["user_id"])
                if sub:
                    plan_name = SUBSCRIPTION_PLANS[sub['plan_type']]['name']
                    st.markdown(f"**Plan:** {plan_name}")
                
                st.divider()
                
                # Admin Dashboard (only for admins)
                if st.session_state.get('is_admin', False):
                    if st.button("🔧 Admin Dashboard", key="admin_dashboard_btn", use_container_width=True):
                        # Close other modals when opening admin dashboard
                        st.session_state.show_profile = False
                        st.session_state.show_settings = False
                        st.session_state.show_subscriptions = False
                        st.session_state.show_admin = True
                        st.rerun()
                    st.divider()
                
                if st.button("👤 My Profile", key="profile_btn", use_container_width=True):
                    # Close other modals
                    st.session_state.show_admin = False
                    st.session_state.show_settings = False
                    st.session_state.show_subscriptions = False
                    st.session_state.show_profile = True
                    st.rerun()
                
                if st.button("💎 Subscriptions", key="subscriptions_btn", use_container_width=True):
                    # Close other modals
                    st.session_state.show_admin = False
                    st.session_state.show_profile = False
                    st.session_state.show_settings = False
                    st.session_state.show_subscriptions = True
                    st.rerun()
                
                if st.button("⚙️ Settings", key="settings_btn_dropdown", use_container_width=True):
                    # Close other modals
                    st.session_state.show_admin = False
                    st.session_state.show_profile = False
                    st.session_state.show_subscriptions = False
                    st.session_state.show_settings = True
                    st.rerun()
                
                st.divider()
                
                if st.button("🚪 Logout", key="logout_btn", use_container_width=True, type="primary"):
                    # Reset ALL session state on logout
                    st.session_state.authenticated = False
                    st.session_state.user_id = None
                    st.session_state.active_user = None
                    st.session_state.is_admin = False
                    st.session_state.current_conversation = None
                    st.session_state.messages = []
                    st.session_state.show_admin = False
                    st.session_state.show_profile = False
                    st.session_state.show_settings = False
                    st.session_state.show_subscriptions = False
                    st.rerun()

def render_profile():
    if st.session_state.show_profile and st.session_state.active_user:
        user = st.session_state.active_user
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("👤 My Profile")
        with col2:
            if st.button("✖️ Close", key="close_profile_btn"):
                st.session_state.show_profile = False
                st.rerun()
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(user["avatar_url"], width=150)
        
        with col2:
            st.markdown(f"### {user['display_name']}")
            st.markdown(f"**Username:** {user['username']}")
            st.markdown(f"**Email:** {user.get('email', 'N/A')}")
            
            # Show subscription
            sub = get_user_subscription(user["user_id"])
            if sub:
                plan_info = SUBSCRIPTION_PLANS[sub['plan_type']]
                st.markdown(f"**Subscription:** {plan_info['name']} (${plan_info['price']}/month)")
            else:
                st.markdown("**Subscription:** Free")
        
        st.divider()
        
def render_sidebar():
    with st.sidebar:
        st.title("💬 Chat History")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        search_term = st.text_input("🔍 Search", key="search_conv", label_visibility="collapsed")
        
        if st.session_state.authenticated and st.session_state.active_user:
            conversations = get_user_conversations(st.session_state.active_user["user_id"])
            
            if search_term:
                conversations = [c for c in conversations if search_term.lower() in c["title"].lower()]
            
            if conversations:
                grouped_convs = group_conversations_by_date(conversations)
                
                for group_name, group_convs in grouped_convs.items():
                    if group_convs:
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
                st.info("No conversations yet!")

def render_settings():
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
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if authenticate_user(user['email'], current_password):
                        update_user_password(user_id, new_password)
                        st.success("Password updated!")
                    else:
                        st.error("Current password is incorrect")
        
        st.divider()
        
        if st.button("✖️ Close Settings"):
            st.session_state.show_settings = False
            st.rerun()

def render_chat():
    """Render main chat interface (No API key setup needed)"""
    
    # Check for admin dashboard
    if st.session_state.get('show_admin', False):
        render_admin_dashboard()
        return
    
    if st.session_state.show_settings:
        render_settings()
        return
    
    if st.session_state.get('show_profile', False):
        render_profile()
        return
    
    if st.session_state.get('show_subscriptions', False):
        render_subscription_page()
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
        st.info("**Active**")
    
    st.divider()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        settings = get_user_settings(st.session_state.active_user["user_id"])
        
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
                        settings
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                else:
                    full_response = call_ai_model(
                        api_messages,
                        st.session_state.selected_model,
                        settings,
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
    
    apply_theme()
    init_database()
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    
    if "show_admin" not in st.session_state:
        st.session_state.show_admin = False
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    if "active_user" not in st.session_state:
        st.session_state.active_user = None
    
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False
    
    if "show_password_recovery" not in st.session_state:
        st.session_state.show_password_recovery = False
    
    if "recovery_user_info" not in st.session_state:
        st.session_state.recovery_user_info = None
    
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    if "show_profile" not in st.session_state:
        st.session_state.show_profile = False
    
    if "show_subscriptions" not in st.session_state:
        st.session_state.show_subscriptions = False
    
    if "theme" not in st.session_state:
        st.session_state.theme = 'light'
    
    # Authentication flow
    if not st.session_state.authenticated:
        if st.session_state.show_password_recovery:
            render_password_recovery()
        elif st.session_state.show_signup:
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
