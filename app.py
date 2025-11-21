import os
import json

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from ml_model import predict_churn
from data_pipeline import get_summary

# --- Load .env & tạo client Groq ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"  # model miễn phí trên Groq


# ====== HÀM GỌI GROQ CHAT (DÙNG CHO KHUNG CHAT) ======
def groq_chat(messages):
    """
    messages: list[{"role": "user"|"assistant"|"system", "content": str}]
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content


# ====== CẤU HÌNH GIAO DIỆN STREAMLIT ======
st.set_page_config(
    page_title="AI Manager Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("AI Manager Assistant")
st.caption(
    "AI Chatbot with Data Analysis and Churn Prediction Features"
)

# Khởi tạo lịch sử chat trong session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm your AI manager assistant. "
                "How can I help you today?"
            ),
        }
    ]


# ====== SIDEBAR: DATA ANALYSIS & PREDICTION ======
st.sidebar.header("📊 Data tools")

# --- Nút xem summary dataset ---
if st.sidebar.button("Show dataset summary"):
    try:
        summary = get_summary()
        st.sidebar.subheader("Dataset summary")
        st.sidebar.json(summary)
    except Exception as e:
        st.sidebar.error(f"Error while summarizing dataset: {e}")

st.sidebar.markdown("---")

# --- Form dự đoán churn ---
st.sidebar.subheader("🔮 Predict customer churn")

with st.sidebar.form("churn_form"):
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0.0,
        max_value=1000.0,
        value=2.0,
        step=1.0
    )
    contract = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"]
    )
    internet = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic"]
    )
    monthly = st.number_input(
        "Monthly charges",
        min_value=0.0,
        max_value=10000.0,
        value=85.5,
        step=0.1
    )

    submit_churn = st.form_submit_button("Predict churn")

if submit_churn:
    try:
        prob = predict_churn(tenure, contract, internet, monthly)
        if isinstance(prob, dict):
            prob = prob.get("churn_probability", 0.0)  # Default to 0.0 if key is missing
        st.sidebar.success(f"Churn probability: {prob:.2%}")
    except Exception as e:
        st.sidebar.error(f"Error while predicting churn: {e}")


# ====== MAIN AREA: KHUNG CHAT ======
st.subheader("💬 Chat")

# Hiển thị lịch sử chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập chat
user_input = st.chat_input("Type your message here...")

if user_input:
    # Lưu tin nhắn user
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Hiển thị tin nhắn user
    with st.chat_message("user"):
        st.markdown(user_input)

    # ====== ROUTING CHO CHAT – KIỂM TRA TASK 2/3 ======
    text = user_input.lower()

    # Nếu user muốn tóm tắt dataset → gọi get_summary()
    if "summarise" in text or "summarize" in text or "summary" in text:
        try:
            summary = get_summary()
            assistant_reply = (
                "Here is the dataset summary:\n\n"
                + json.dumps(summary, indent=2)
            )
        except Exception as e:
            assistant_reply = (
                f"Sorry, an error occurred while summarizing the dataset: {e}"
            )

    # Nếu user muốn dự đoán churn → gọi ML model
    elif "predict churn" in text or "churn prediction" in text:
        try:
            assistant_reply = predict_churn(tenure, contract, internet, monthly)
        except Exception as e:
            assistant_reply = (
                f"Sorry, an error occurred while predicting churn: {e}"
            )

    # Nếu không → chat như bình thường
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for a manager. "
                    "You can chat in English, explain business concepts, "
                    "and help with management decisions."
                ),
            }
        ]
        for m in st.session_state.chat_history:
            messages.append({"role": m["role"], "content": m["content"]})

        try:
            assistant_reply = groq_chat(messages)
        except Exception as e:
            assistant_reply = (
                f"Sorry, an error occurred while calling Groq API: {e}"
            )

    # Lưu reply vào lịch sử
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_reply}
    )

    # Hiển thị reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
