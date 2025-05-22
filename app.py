import gradio as gr
import pandas as pd
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load models once
tox_model = Detoxify('multilingual')
ai_tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
ai_model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")

# Thresholds
TOXICITY_THRESHOLD = 0.7
AI_THRESHOLD = 0.5

def detect_ai(text):
    with torch.no_grad():
        inputs = ai_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        logits = ai_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    return round(probs[1], 4)  # Probability of AI-generated

def classify_comments(comment_list):
    results = tox_model.predict(comment_list)
    df = pd.DataFrame(results, index=comment_list).round(4)
    df.columns = [col.replace("_", " ").title().replace(" ", "_") for col in df.columns]
    df.columns = [col.replace("_", " ") for col in df.columns]
    df["⚠️ Warning"] = df.apply(
        lambda row: "⚠️ High Risk" if any(score > TOXICITY_THRESHOLD for score in row) else "✅ Safe",
        axis=1
    )
    df["🧪 AI Probability"] = [detect_ai(c) for c in df.index]
    df["🧪 AI Detection"] = df["🧪 AI Probability"].apply(
        lambda x: "🤖 Likely AI" if x > AI_THRESHOLD else "🧍 Human"
    )
    return df

def run_classification(text_input, csv_file):
    comment_list = []

    if text_input.strip():
        comment_list += [c.strip() for c in text_input.strip().split('\n') if c.strip()]

    if csv_file:
        df = pd.read_csv(csv_file.name)
        if 'comment' not in df.columns:
            return "CSV must contain a 'comment' column.", None
        comment_list += df['comment'].astype(str).tolist()

    if not comment_list:
        return "Please provide comments via text or CSV.", None

    df = classify_comments(comment_list)
    csv_data = df.copy()
    csv_data.insert(0, "Comment", df.index)
    return df, ("toxicity_predictions.csv", csv_data.to_csv(index=False).encode())

# Build the Gradio UI
with gr.Blocks(title="🌍 Toxic Comment & AI Detector") as app:
    gr.Markdown("## 🌍 Toxic Comment & AI Detector")
    gr.Markdown("Detects multilingual toxicity and whether a comment is AI-generated. Paste comments or upload a CSV.")

    with gr.Row():
        text_input = gr.Textbox(lines=8, label="💬 Paste Comments (one per line)")
        csv_input = gr.File(label="📥 Upload CSV (must have 'comment' column)")

    submit_button = gr.Button("🔍 Analyze Comments")
    output_table = gr.Dataframe(label="📊 Prediction Results")
    download_button = gr.File(label="📤 Download CSV")

    submit_button.click(fn=run_classification, inputs=[text_input, csv_input], outputs=[output_table, download_button])

if __name__ == "__main__":
    app.launch()