import streamlit as st
from groq import Groq
import json
import os


CHAT_HISTORY_FILE = "chat_history.json"


def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(st.session_state.messages, file, indent=4)


if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history()


client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))


if "llm" not in st.session_state:
    st.session_state["llm"] = ""


st.header("Multi Model Chatbot Playground", divider="orange", anchor=False)
st.title(":orange[Chat App]", anchor=False)


st.sidebar.title("Parameters")


def reset_chat():
    st.session_state.messages = []
    save_chat_history()  # Save the cleared history
    st.toast(f"Model Selected: {st.session_state.llm}", icon="🤖")

st.session_state.llm = st.sidebar.selectbox("Select Model", [
    "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"
], index=0, on_change=reset_chat)


temp = st.sidebar.slider("Temperature", 0.0, 2.0, value=1.0)
max_tokens = st.sidebar.slider("Max Tokens", 0, 8192, value=1024)
stream = st.sidebar.toggle("Stream", value=True)
json_mode = st.sidebar.toggle("JSON Mode", help="You must also ask the model to return JSON.")


with st.sidebar.expander("Advanced"):
    top_p = st.slider("Top P", 0.0, 1.0, help="It's not recommended to alter both the temperature and the top-p.")
    stop_seq = st.text_input("Stop Sequence")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history() 


    with st.chat_message("user"):
        st.write(prompt)


    with st.chat_message("assistant"):
        response_text = st.empty()
        full_response = ""

        try:
            completion = client.chat.completions.create(
                model=st.session_state.llm or "llama3-8b-8192",
                messages=st.session_state.messages,
                stream=stream,
                temperature=temp,
                max_tokens=max_tokens,
                response_format={"type": "json_format"} if json_mode else {"type": "text"},
                stop=stop_seq,
                top_p=top_p
            )


            if stream:
                for chunk in completion:
                    full_response += chunk.choices[0].delta.content or ""
                    response_text.write(full_response)
            else:
                full_response = completion.choices[0].message.content
                response_text.write(full_response)


            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat_history() 

        except Exception as e:
            st.error(f"Error: {e}")
