import os
import json

from dotenv import load_dotenv
from groq import Groq

from ml_model import predict_churn
from data_pipeline import get_summary

# --- Load biến môi trường (.env) ---
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

MODEL_NAME = "llama-3.3-70b-versatile"  # hoặc model khác mà Groq cho


def handle_predict_churn() -> str:
    """
    Hỏi user từng thông tin rồi gọi mô hình ML predict_churn.
    Đây chính là phần 'function calling' từ chatbot tới ML model.
    """
    try:
        tenure = float(input("Bot: Enter tenure in months: "))
        contract = input(
            "Bot: Enter contract (Month-to-month / One year / Two year): "
        ).strip()
        internet = input(
            "Bot: Enter internet type (DSL / Fiber optic): "
        ).strip()
        monthly = float(input("Bot: Enter monthly charges: "))

        prob = predict_churn(tenure, contract, internet, monthly)
        return (
            f"Based on the model, the churn probability is about {prob:.2%}."
        )

    except Exception as e:
        return f"Something went wrong while predicting churn: {e}"


def handle_summary() -> str:
    """
    Gọi hàm get_summary() trong data_pipeline để phân tích dataset.
    """
    try:
        summary = get_summary()
        # In đẹp dạng JSON
        return (
            "Here is the statistical summary of the dataset:\n"
            + json.dumps(summary, indent=2)
        )
    except Exception as e:
        return f"Something went wrong while summarizing the dataset: {e}"


def handle_general_chat(user_message: str) -> str:
    """
    Chat bình thường với Groq (Task 1: AI chatbot).
    """
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for a manager. "
                    "You can chat in English and give helpful answers."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    return (
        chat_completion.choices[0].message.content or ""
    )  # Ensure return is always a string


def chat_with_assistant(user_message: str) -> str:
    """
    Router chính:
      - Nếu user muốn dự đoán churn -> gọi predict_churn (ML model)
      - Nếu user muốn phân tích / summary data -> gọi get_summary
        (data pipeline)
      - Còn lại -> chat với Groq như bình thường
    Đây là phần 'assistant feature (function calling)' của bài CW.
    """
    text = user_message.lower()

    # Từ khóa kích hoạt function ML model
    if "predict churn" in text or "churn prediction" in text:
        return handle_predict_churn()

    # Từ khóa kích hoạt data pipeline
    if "summary" in text or "summarise" in text or "analyze dataset" in text:
        return handle_summary()

    # Mặc định: chat LLM
    return handle_general_chat(user_message)


def main():
    print("AI Manager Assistant using Groq (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        try:
            reply = chat_with_assistant(user_input)
            print("Bot:", reply)
        except Exception as e:
            print("Bot: An error occurred:", e)


if __name__ == "__main__":
    main()
