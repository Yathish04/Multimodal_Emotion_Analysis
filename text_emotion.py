from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from googletrans import Translator
import torch
import nest_asyncio
import asyncio

# Patch asyncio to allow nested event loops (needed for Colab / Jupyter)
nest_asyncio.apply()

# Initialize translator and model
translator = Translator()

MODEL = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Emotion labels for this model
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

async def translate_to_english_async(text):
    """Translate text asynchronously (safe for Colab/Jupyter)."""
    translated = translator.translate(text, dest='en')
    # If it's a coroutine, await it
    if asyncio.iscoroutine(translated):
        translated = await translated
    return translated.text

def translate_to_english(text):
    """Safe wrapper for both sync and async environments."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Already running (e.g., in Jupyter)
        return loop.run_until_complete(translate_to_english_async(text))
    else:
        return asyncio.run(translate_to_english_async(text))

def predict_emotion_english(text):
    """Predict emotion for English text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0].detach().numpy()
    probs = softmax(scores)
    emotion = labels[probs.argmax()]
    confidence = probs.max()
    return emotion, confidence

# --- Interactive loop ---
print("🌐 Universal Emotion Detector (via Translation to English)")
print("Type any text (in any language) to detect its emotion.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    text = input("Enter text: ").strip()
    if text.lower() in ["quit", "exit"]:
        print("👋 Exiting the emotion detector. Goodbye!")
        break
    if not text:
        print("⚠️ Please enter some text!")
        continue

    translated = translate_to_english(text)
    print(f"🔁 Translated to English: \"{translated}\"")

    emotion, confidence = predict_emotion_english(translated)
    print(f"→ Emotion: {emotion} (Confidence: {confidence:.2f})\n")
