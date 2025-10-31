"""
Multi-Model AI Chatbot with Conversation History
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

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            avatar_url TEXT,
            api_key TEXT,
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
    
    # Create default user if none exists
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (username, display_name, avatar_url, api_key)
            VALUES (?, ?, ?, ?)
        """, ("default_user", "Default User", "https://api.dicebear.com/7.x/avataaars/svg?seed=default", None))
        
        user_id = cursor.lastrowid
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt)
            VALUES (?, ?)
        """, (user_id, "You are a helpful AI assistant."))
    
    conn.commit()
    conn.close()

def get_all_users() -> List[Dict]:
    """Get all users from database"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, display_name, avatar_url, api_key FROM users")
    users = [
        {"user_id": row[0], "username": row[1], "display_name": row[2], "avatar_url": row[3], "api_key": row[4]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return users

def create_user(username: str, display_name: str, avatar_seed: str, api_key: str = None) -> int:
    """Create a new user"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    avatar_url = f"https://api.dicebear.com/7.x/avataaars/svg?seed={avatar_seed}"
    
    try:
        cursor.execute("""
            INSERT INTO users (username, display_name, avatar_url, api_key)
            VALUES (?, ?, ?, ?)
        """, (username, display_name, avatar_url, api_key))
        
        user_id = cursor.lastrowid
        
        # Create default settings
        cursor.execute("""
            INSERT INTO settings (user_id, system_prompt)
            VALUES (?, ?)
        """, (user_id, "You are a helpful AI assistant."))
        
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()

def update_user_api_key(user_id: int, api_key: str):
    """Update user's API key"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET api_key = ? WHERE user_id = ?", (api_key, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id: int):
    """Delete a user and all associated data"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Delete messages
    cursor.execute("""
        DELETE FROM messages WHERE conversation_id IN 
        (SELECT conversation_id FROM conversations WHERE user_id = ?)
    """, (user_id,))
    
    # Delete conversations
    cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
    
    # Delete settings
    cursor.execute("DELETE FROM settings WHERE user_id = ?", (user_id,))
    
    # Delete user
    cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    
    conn.commit()
    conn.close()

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
    return {}

def update_user_settings(user_id: int, settings: Dict):
    """Update user settings"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
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
        "X-Title": "Multi-Model AI Chatbot"
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
# UI FUNCTIONS
# ========================================

def render_api_key_setup():
    """Render API key setup screen"""
    st.title("🔑 API Key Setup Required")
    
    st.markdown("""
    ### Welcome to Multi-Model AI Chatbot!
    
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
                # Update current user's API key
                update_user_api_key(st.session_state.active_user["user_id"], api_key)
                st.session_state.active_user["api_key"] = api_key
                st.success("API key saved! Redirecting...")
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
        
        # User profile section
        if st.session_state.active_user:
            user = st.session_state.active_user
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(user["avatar_url"], width=50)
            with col2:
                st.write(f"**{user['display_name']}**")
                if st.button("⚙️ Settings", key="settings_btn", use_container_width=True):
                    st.session_state.show_settings = True
                    st.rerun()
        
        st.divider()
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Search conversations
        search_term = st.text_input("🔍 Search conversations", key="search_conv")
        
        # Display conversations
        if st.session_state.active_user:
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
        
        tabs = st.tabs(["API Key", "Chat Behavior", "User Profile", "Account Management"])
        
        # API Key Tab
        with tabs[0]:
            st.subheader("🔑 OpenRouter API Key")
            
            current_key = st.session_state.active_user.get("api_key", "")
            
            if current_key:
                st.success("✅ API key is configured")
                masked_key = current_key[:10] + "..." + current_key[-4:] if len(current_key) > 14 else "***"
                st.code(masked_key)
            else:
                st.warning("⚠️ No API key configured")
            
            st.divider()
            
            new_api_key = st.text_input(
                "Update API Key:",
                type="password",
                placeholder="sk-or-v1-...",
                help="Get your key from https://openrouter.ai/keys"
            )
            
            if st.button("💾 Update API Key", type="primary"):
                if new_api_key and new_api_key.startswith("sk-or-"):
                    update_user_api_key(user_id, new_api_key)
                    st.session_state.active_user["api_key"] = new_api_key
                    st.success("API key updated!")
                    st.rerun()
                else:
                    st.error("Please enter a valid OpenRouter API key")
            
            st.divider()
            
            st.markdown("""
            **Get more credits:**
            - Visit [OpenRouter](https://openrouter.ai/)
            - Add payment method for pay-as-you-go
            - Monitor usage in your dashboard
            """)
        
        # Chat Behavior Tab
        with tabs[1]:
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
        
        # User Profile Tab
        with tabs[2]:
            st.subheader("👤 Profile Information")
            
            user = st.session_state.active_user
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(user["avatar_url"], width=100)
            with col2:
                st.write(f"**Username:** {user['username']}")
                st.write(f"**Display Name:** {user['display_name']}")
            
            st.info("To change profile information, create a new account.")
        
        # Account Management Tab
        with tabs[3]:
            st.subheader("👥 Manage Accounts")
            
            # List all users
            all_users = get_all_users()
            
            st.write("**Switch Account:**")
            for user in all_users:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.image(user["avatar_url"], width=40)
                with col2:
                    is_current = user["user_id"] == st.session_state.active_user["user_id"]
                    button_label = f"✓ {user['display_name']}" if is_current else user['display_name']
                    if st.button(
                        button_label,
                        key=f"switch_{user['user_id']}",
                        disabled=is_current,
                        use_container_width=True
                    ):
                        st.session_state.active_user = user
                        st.session_state.current_conversation = None
                        st.session_state.messages = []
                        st.rerun()
                with col3:
                    if len(all_users) > 1:
                        if st.button("🗑️", key=f"delete_user_{user['user_id']}"):
                            if user["user_id"] != st.session_state.active_user["user_id"]:
                                delete_user(user["user_id"])
                                st.success(f"Deleted user: {user['display_name']}")
                                st.rerun()
                            else:
                                st.error("Cannot delete active user. Switch accounts first.")
            
            st.divider()
            
            # Create new account
            with st.expander("➕ Create New Account"):
                new_username = st.text_input("Username", key="new_username")
                new_display_name = st.text_input("Display Name", key="new_display")
                new_avatar_seed = st.text_input("Avatar Seed", value=new_username, key="new_avatar")
                new_user_api_key = st.text_input("API Key (optional)", type="password", key="new_api_key")
                
                if st.button("Create Account", type="primary"):
                    if new_username and new_display_name:
                        user_id = create_user(new_username, new_display_name, new_avatar_seed, new_user_api_key or None)
                        if user_id > 0:
                            st.success(f"Created account: {new_display_name}")
                            # Switch to new user
                            new_user = next(u for u in get_all_users() if u["user_id"] == user_id)
                            st.session_state.active_user = new_user
                            st.rerun()
                        else:
                            st.error("Username already exists!")
                    else:
                        st.error("Please fill username and display name")
        
        st.divider()
        
        if st.button("✖️ Close Settings"):
            st.session_state.show_settings = False
            st.rerun()

def render_chat():
    """Render main chat interface"""
    # Check if API key is configured
    if not st.session_state.active_user.get("api_key"):
        render_api_key_setup()
        return
    
    if st.session_state.show_settings:
        render_settings()
        return
    
    # Header with model selector
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.title("🤖 AI Chatbot")
    
    with col2:
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
    
    with col3:
        st.write("")
        st.write("")
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
        
        # Create conversation if needed
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
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
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

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    st.set_page_config(
        page_title="Multi-Model AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_database()
    
    # Initialize session state
    if "active_user" not in st.session_state:
        users = get_all_users()
        st.session_state.active_user = users[0] if users else None
    
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    # Render UI
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
