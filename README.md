# 🌍 Toxic Comment & AI Detector

A Hugging Face Space that analyzes comments for:

- 🧪 AI-generated text detection (`roberta-base-openai-detector`)
- ☣️ Toxicity detection (multilingual support via `Detoxify`)
- 📥 CSV input support with `comment` column
- 📊 Result table with:
  - Toxicity metrics (toxicity, insult, threat, etc.)
  - AI probability
  - Human/AI classification
- 📤 CSV output download

---

## 🚀 Live Demo

👉 **Try it here:** [https://huggingface.co/spaces/KavinduHansaka/toxic-comment-ai-detector](https://huggingface.co/spaces/KavinduHansaka/toxic-comment-ai-detector)

---

## 🧠 Models Used

| Purpose            | Model                                     |
|-------------------|--------------------------------------------|
| AI Detection       | `openai-community/roberta-base-openai-detector` |
| Toxicity Detection | `unitary/toxic-bert` via `Detoxify` (multilingual) |

