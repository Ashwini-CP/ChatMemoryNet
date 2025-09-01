
import streamlit as st
import requests

st.set_page_config(page_title='ChatMemoryNet', page_icon='🧠', layout='wide')
st.title('🧠 ChatMemoryNet — Health Memory Bot (Tanglish + English)')

backend_url = st.sidebar.text_input('Backend URL', 'http://localhost:8000')
user_id = st.sidebar.text_input('User ID', 'demo')

col1, col2 = st.columns([2,1])

with col2:
    if st.button('Rebuild Index'):
        r = requests.post(f'{backend_url}/rebuild')
        if r.ok:
            st.success(f"Rebuilt embeddings for {r.json().get('count', '?')} symptoms")
        else:
            st.error('Backend unreachable')

with col1:
    st.write('Type your symptoms below. If you use **Tanglish**, reply will be Tanglish; English → English.')
    if 'chat' not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input('You:', value='', key='chat_input')
    if st.button('Send') and msg.strip():
        payload = {'user_id': user_id, 'message': msg.strip()}
        r = requests.post(f'{backend_url}/chat', json=payload)
        if r.ok:
            data = r.json()
            st.session_state.chat.append(('You', msg.strip()))
            st.session_state.chat.append(('Bot', data['reply']))
        else:
            st.error('Backend error')

    for role, text in st.session_state.chat[-12:]:
        st.markdown(f'**{role}:** {text}')

st.divider()
st.subheader('📈 Memory Graph (raw JSON)')
if st.button('Refresh Graph JSON'):
    rg = requests.get(f'{backend_url}/graph')
    if rg.ok:
        st.json(rg.json())
    else:
        st.error('Failed to fetch graph')
