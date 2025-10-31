# 🤖 Multi-Model AI Chatbot

A powerful, customizable AI chatbot that supports multiple leading AI models including GPT-4o, Claude Sonnet 4, Gemini 2.0 Flash, Grok-3, and more.

## ✨ Features

- **Multiple AI Models**: Switch between OpenAI, Anthropic Claude, Google Gemini, xAI Grok, and DeepSeek
- **Persistent Chat History**: All conversations saved with SQLite database
- **User Profile Management**: Create and manage multiple user accounts
- **Customizable Settings**: 
  - Adjust temperature, max tokens, tone, and verbosity
  - Custom system prompts
  - Stream or batch responses
- **Conversation Management**: 
  - Create, view, search, and delete conversations
  - Export chat history
  - Organized sidebar navigation

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-chatbot.git
cd ai-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
