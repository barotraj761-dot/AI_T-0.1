import streamlit as st
from transformers import pipeline
import torch

# Page Configuration
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Coding & Chat Assistant")

# Model ko cache kar rahe hain taaki har user input par model baar-baar reload na ho
@st.cache_resource
def load_assistant():
    return pipeline(
        "text-generation",
        model="microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
        torch_dtype="auto"
    )

with st.spinner("AI Assistant load ho raha hai... (Pehli baar thoda time lag sakta hai)"):
    assistant_pipe = load_assistant()

# Chat History initialize karna
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, smart, and friendly AI Assistant. You can write code, solve problems, and chat nicely."}
    ]

# UI par purane messages display karna (System prompt ko chhod kar)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User Input Widget
if user_input := st.chat_input("Apna message ya prompt likhein..."):
    # User message display aur history me add karna
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Model reply generate karna
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            outputs = assistant_pipe(
                st.session_state.messages,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True
            )
            # Response extract karna
            assistant_reply = outputs[0]['generated_text'][-1]['content']
            st.markdown(assistant_reply)

    # Assistant reply ko history me save karna
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
