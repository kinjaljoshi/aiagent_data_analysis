import sys
import os
import pandas as pd
import gradio as gr
import argparse

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from query_workflow import process_query

__all__ = ["stream_to_gradio", "GradioUI"]

chat_history = []

def stream_to_gradio(user_query):
    global chat_history
    if not user_query:
        return chat_history, "", gr.update(visible=False)

    response = process_query(user_query)

    if isinstance(response, pd.DataFrame):
        chat_history.append(("🧑 " + user_query, "🤖 Here's the data ⬇️"))
        return chat_history, "", gr.update(value=response, visible=True)
    else:
        chat_history.append(("🧑 " + user_query, "🤖 " + response))
        return chat_history, "", gr.update(visible=False)

def clear_chat():
    global chat_history
    chat_history = []
    return [], "", gr.update(visible=False)

def GradioUI():
    with gr.Blocks(title="Personalized Query Assistant") as demo:
        gr.Markdown("## 💬 Personalized Query Assistant")

        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Ask your question...", show_label=False)

        df_output = gr.Dataframe(visible=False)
        clear_btn = gr.Button("🧹 Clear Chat")

        user_input.submit(stream_to_gradio, inputs=user_input, outputs=[chatbot, user_input, df_output])
        clear_btn.click(fn=clear_chat, outputs=[chatbot, user_input, df_output])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on")
    args = parser.parse_args()

    ui = GradioUI()
    ui.launch(server_port=args.port)
