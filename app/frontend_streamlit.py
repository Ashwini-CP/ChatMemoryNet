import streamlit as st
import requests

# ================================
# 🔧 Page Config
# ================================
st.set_page_config(page_title='ChatMemoryNet', page_icon='🧠', layout='wide')

st.title("🧠 ChatMemoryNet — Health Memory Bot (Tanglish)")

# ================================
# 🔧 Sidebar
# ================================
backend_url = st.sidebar.text_input('🔗 Backend URL', 'http://localhost:8000')
user_id = st.sidebar.text_input('👤 User ID', 'demo')

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
chat_container = st.container()
with chat_container:
    for role, text in st.session_state.chat[-20:]:
        if role == "user":
            st.markdown(f"<div style='background:#007AFF;color:white;padding:10px;border-radius:10px;text-align:right;margin:5px'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#f0f0f0;color:black;padding:10px;border-radius:10px;text-align:left;margin:5px'>{text}</div>", unsafe_allow_html=True)

# ================================
# 📊 Memory Graph Section
# ================================
st.subheader("📊 Memory Graph")

col_json, col_viz = st.columns([1, 2])

with col_json:
    st.markdown("### 📝 JSON View")
    if st.button('🔄 Refresh JSON'):
        try:
            rg = requests.get(f'{backend_url}/graph')
            if rg.ok:
                st.json(rg.json())
            else:
                st.error('❌ Failed to fetch graph JSON')
        except Exception as e:
            st.error(f'⚠️ Connection error: {e}')

with col_viz:
    st.markdown("### 🌐 Interactive Graph")
    try:
        gv = requests.get(f'{backend_url}/graphviz')
        if gv.ok:
            st.components.v1.html(gv.text, height=550, scrolling=True)
        else:
            st.error('❌ Failed to fetch graph visualization')
    except Exception as e:
        st.error(f'⚠️ Connection error: {e}')
