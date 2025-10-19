
#  LoRA Fine-Tuning — DistilRoBERTa on IMDB Sentiment Dataset

This project demonstrates how to **efficiently fine-tune a transformer model** using **LoRA (Low-Rank Adaptation)** on the **IMDB Movie Reviews dataset** for binary sentiment classification.


---

##  Overview

| Component | Details |
|------------|----------|
| **Base Model** | [`distilroberta-base`](https://huggingface.co/distilroberta-base) |
| **Fine-tuning Method** | [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) via `PEFT` |
| **Dataset** | [IMDB Reviews](https://huggingface.co/datasets/imdb) |
| **Frameworks** | 🤗 Transformers • PEFT • Datasets • Accelerate |
| **Training Env** | Google Colab (T4 GPU) |
| **Goal** | Sentiment Classification (Positive / Negative) |
| **Accuracy (demo)** | ~87% after 2 epochs on subset data |

---

##  Why LoRA?

Traditional fine-tuning updates **100% of model weights**, which is expensive for large models.  
**LoRA** introduces small, trainable rank-decomposition matrices into existing layers — keeping the base model frozen.

| Aspect | Full Fine-Tuning | LoRA Fine-Tuning |
|:--|:--|:--|
| Trainable Params | 100% | ~1–2% |
| GPU Memory | High |  Low |
| Training Speed | Slow |  Faster |
| Reusability | One model per task | One base + many adapters |
| Ideal For | Big infra | Colab / small GPU setups |

This makes LoRA **perfect for learners, researchers, and startups** doing domain-specific tuning efficiently.

---

##  Learning Outcomes

- Understand parameter-efficient fine-tuning (PEFT) concepts.  
- Implement LoRA with Hugging Face `transformers` and `peft`.  
- Fine-tune and evaluate a model on IMDB sentiment data.  
- Merge LoRA adapters into the base model for deployment.  
- Push the final model to Hugging Face Hub or use completely offline.

---


---

##  Quick Start (Offline or Online)

### 🔹 Use locally
```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model="./distilroberta-lora-merged", tokenizer="./distilroberta-lora-merged", local_files_only=True)
print(sentiment("This movie was absolutely amazing!"))
````

### 🔹 Use from Hugging Face Hub

```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model="RafiCoding/lora-finetune-distilroberta-imdb")
sentiment("A beautifully directed film with emotional depth!")
```

---

##  Results Summary

| Metric       |                Score |
| :----------- | -------------------: |
| Accuracy     |                0.874 |
| F1-Score     |                0.874 |
| Eval Runtime | ~6 min (on Colab T4) |

---

##  Author

**Sadman Sakib Rafi**
Machine Learning Engineer / AI Developer
📧 [sadmansakibrafi.hey@gmail.com]
---
