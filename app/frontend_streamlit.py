import streamlit as st
import requests

# ================================
# 🔧 Page Config
# ================================
st.set_page_config(page_title='ChatMemoryNet', page_icon='🧠', layout='wide')
st.markdown(
    """
    <style>
    /* 🌈 Gradient background */
    body {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Chat container styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.85);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in-out;
    }

    /* Chat bubble styles */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 20px;
        margin: 8px 0;
        max-width: 70%;
        font-size: 16px;
        line-height: 1.5;
        word-wrap: break-word;
        animation: slideUp 0.4s ease;
    }
    .user-bubble {
        background: linear-gradient(135deg, #007AFF, #00c6ff);
        color: white;
        margin-left: auto;
        text-align: right;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .bot-bubble {
        background: linear-gradient(135deg, #fdfbfb, #ebedee);
        color: black;
        margin-right: auto;
        text-align: left;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 ChatMemoryNet — Health Memory Bot (Tanglish + English)")

# ================================
# 🔧 Sidebar
# ================================
backend_url = st.sidebar.text_input('🔗 Backend URL', 'http://localhost:8000')
user_id = st.sidebar.text_input('👤 User ID', 'demo')

if st.sidebar.button('♻️ Rebuild Index'):
    r = requests.post(f'{backend_url}/rebuild')
    if r.ok:
        st.sidebar.success(f"Rebuilt embeddings for {r.json().get('count', '?')} symptoms")
    else:
        st.sidebar.error('Backend unreachable')

# ================================
# 💬 Chat Section
# ================================
st.subheader("💬 Chat")

if 'chat' not in st.session_state:
    st.session_state.chat = []

col1, col2 = st.columns([4, 1])
with col1:
    msg = st.text_input('You:', value='', key='chat_input')
with col2:
    send = st.button('🚀 Send')

if send and msg.strip():
    payload = {'user_id': user_id, 'message': msg.strip()}
    try:
        r = requests.post(f'{backend_url}/chat', json=payload)
        if r.ok:
            data = r.json()
            st.session_state.chat.append(('user', msg.strip()))
            st.session_state.chat.append(('bot', data['reply']))
        else:
            st.error('⚠️ Backend error')
    except Exception as e:
        st.error(f'❌ Connection error: {e}')

# ================================
# 🖼️ Chat History Display
# ================================
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for role, text in st.session_state.chat[-20:]:
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f"<div class='chat-bubble {bubble_class}'>{text}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ================================
# 📊 Memory Graph Section
# ================================
with st.expander("📈 Memory Graph (raw JSON)", expanded=False):
    if st.button('🔄 Refresh Graph JSON'):
        try:
            rg = requests.get(f'{backend_url}/graph')
            if rg.ok:
                st.json(rg.json())
            else:
                st.error('❌ Failed to fetch graph')
        except Exception as e:
            st.error(f'⚠️ Connection error: {e}')
