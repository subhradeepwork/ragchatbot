
import gradio as gr
import requests

API_URL = "http://localhost:8000"

def upload_pdf(file):
    if file is None:
        return "Please upload a PDF first."
    with open(file.name, "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": (file.name, f, "application/pdf")})
    return response.json().get("message", "Upload failed.")

def ask_question(message, history):
    response = requests.post(f"{API_URL}/chat", json={"question": message})
    if response.status_code != 200:
        return "Something went wrong!", history
    result = response.json()
    answer = result["answer"]
    sources = "\n\n".join(result["sources"])
    history.append((message, f"{answer}\n\n📚 **Sources:**\n{sources}"))
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("## 📄 RAG Chatbot — Upload PDF and Ask Questions")
    with gr.Row():
        file_uploader = gr.File(label="Upload your PDF")
        upload_btn = gr.Button("Upload")
    upload_status = gr.Textbox(label="Upload Status")
    upload_btn.click(upload_pdf, inputs=[file_uploader], outputs=[upload_status])

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    msg.submit(ask_question, [msg, chatbot], [msg, chatbot])

demo.launch()
