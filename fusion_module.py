import os
import sys
import time
import threading
import queue
import numpy as np
import cv2
import torch
import sounddevice as sd
import soundfile as sf
import tempfile
import librosa
try:
    import speech_recognition as sr
except ImportError:
    sr = None
    print("⚠️ SpeechRecognition not found. Audio-to-Text disabled.")
import traceback

# --- Import Deep Learning Libraries ---
# Wrapped in try/except to handle TensorFlow/Keras version mismatch
try:
    from deepface import DeepFace
except ValueError as e:
    if "tf-keras" in str(e).lower():
         print("\n❌ CRITICAL DEPENDENCY ERROR: 'tf-keras' is missing.")
         print("------------------------------------------------------")
         print("Please run: pip install tf-keras")
         print("------------------------------------------------------\n")
         # sys.exit(1) # Don't exit, just warn
    # raise e
except ImportError:
    print("DeepFace import failed.")
    # sys.exit(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from googletrans import Translator
from speechbrain.inference import foreign_class

# ==========================================
# CONFIGURATION
# ==========================================
CHUNK_DURATION = 3.5       # Duration of audio chunks (seconds)
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01   # Audio RMS threshold to ignore background noise
WEIGHTS = {'face': 0.5, 'voice': 0.3, 'text': 0.2}
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Colors for UI (BGR format)
COLORS = {
    'angry': (0, 0, 255), 'disgust': (147, 20, 255), 'fear': (128, 0, 128),
    'happy': (0, 255, 0), 'sad': (255, 0, 0), 'surprise': (0, 255, 255),
    'neutral': (200, 200, 200)
}

# ==========================================
# 1. TEXT EMOTION (DistilRoberta)
# ==========================================
class TextEmotionAnalyzer:
    def __init__(self):
        print("Loading Text Model...")
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as e:
            print(f"❌ Text Model Load Error: {e}")
            self.model = None

        try:
            self.translator = Translator()
        except Exception as e:
            print(f"⚠️ Translator Load Error: {e}")
            self.translator = None

        self.label_map = {
            'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear', 
            'joy': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprise'
        }

    def predict(self, text):
        if not text or not self.model: return None, None
        try:
            # Detect and translate if not English
            if self.translator:
                try:
                    if self.translator.detect(text).lang != 'en':
                        text = self.translator.translate(text, dest='en').text
                except Exception:
                    pass # Translation failed, use original
        except: pass

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = softmax(outputs.logits[0].detach().numpy())
            result = {l: 0.0 for l in LABELS}
            
            model_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            for i, ml in enumerate(model_labels):
                sl = self.label_map.get(ml)
                if sl: result[sl] = float(probs[i])
                
            top = max(result, key=result.get)
            return result, top
        except Exception as e:
            print(f"Text Predict Error: {e}")
            return None, None

# ==========================================
# 2. VOICE EMOTION (SpeechBrain)
# ==========================================
class VoiceEmotionAnalyzer:
    def __init__(self):
        print("Loading Voice Model...")
        try:
            # Try loading without custom_interface first if it's missing
            self.classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier"
            )
        except Exception as e:
            print(f"⚠️ Voice Model Load Error (Attempt 1): {e}")
            try:
                # Fallback: Try default loading (might download interface.py)
                self.classifier = foreign_class(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    classname="CustomEncoderWav2vec2Classifier"
                )
            except Exception as e2:
                print(f"❌ Voice Model Load Error (Attempt 2): {e2}")
                self.classifier = None

        # IEMOCAP typically maps to: neutral, anger, happiness, sadness
        self.label_map = {'ang': 'angry', 'hap': 'happy', 'sad': 'sad', 'neu': 'neutral', 'exc': 'happy'}

    def predict(self, wav_path):
        if not self.classifier: return None, None
        try:
            # Windows Path Fix
            wav_path = os.path.abspath(wav_path).replace("\\", "/")
            
            # --- Preprocessing: Resample to 16kHz Mono ---
            # This fixes issues with browser audio (often 48kHz/WebM)
            try:
                # Load with librosa (handles resampling & mono mix)
                # Note: On Windows, reading WebM might require ffmpeg. 
                # If librosa fails, we fall back to the original file.
                sig, fs = librosa.load(wav_path, sr=16000, mono=True)
                
                # Save to a clean 16-bit PCM WAV temp file
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                sf.write(tmp.name, sig, 16000, subtype='PCM_16')
                
                predict_path = tmp.name.replace("\\", "/")
            except Exception as e:
                print(f"⚠️ Audio Preprocessing Failed (using original): {e}")
                predict_path = wav_path

            _, _, _, text_lab = self.classifier.classify_file(predict_path)
            
            # Cleanup temp file
            if predict_path != wav_path and os.path.exists(predict_path):
                try: os.remove(predict_path)
                except: pass
            
            detected = text_lab[0] if isinstance(text_lab, list) else text_lab
            std = self.label_map.get(detected, 'neutral')
            
            # Create probability distribution 
            # (High confidence for detected, low for others because SB output is hard label)
            result = {l: 0.05 for l in LABELS} # minimal noise
            result[std] = 0.70                 # dominant score
            
            return result, std
        except Exception as e:
            print(f"Voice Predict Error: {e}")
            return None, None

# ==========================================
# 3. FUSION ENGINE
# ==========================================
class EmotionFusion:
    def __init__(self):
        self.text_eng = TextEmotionAnalyzer()
        self.voice_eng = VoiceEmotionAnalyzer()
        self.rec = sr.Recognizer() if sr else None
        self.lock = threading.Lock()
        self.paused = False
        
        # Shared State
        self.data = {
            'face': {'probs': None, 'top': 'waiting...'},
            'voice': {'probs': None, 'top': 'silent'},
            'text': {'probs': None, 'top': 'empty', 'content': ''}
        }

    def update_face(self, probs):
        with self.lock:
            if probs:
                top = max(probs, key=probs.get)
                self.data['face'] = {'probs': probs, 'top': top}

    def process_audio(self):
        print("🎙️ Audio Loop Started...")
        while True:
            if self.paused:
                time.sleep(0.5)
                continue

            try:
                # 1. Record Audio
                rec = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                rec = rec.flatten()
                
                # 2. RMS Gate (Silence Detection)
                rms = np.sqrt(np.mean(rec**2))
                if rms < SILENCE_THRESHOLD:
                    with self.lock:
                        self.data['voice'] = {'probs': None, 'top': 'silent'}
                    continue

                # 3. Save Temp File
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                sf.write(tmp.name, rec, SAMPLE_RATE)
                
                # 4. Voice Prediction
                v_probs, v_top = self.voice_eng.predict(tmp.name)
                
                # 5. Text Prediction (STT -> NLP)
                transcript = ""
                t_probs, t_top = None, None
                try:
                    if self.rec:
                        with sr.AudioFile(tmp.name) as source:
                            audio = self.rec.record(source)
                            transcript = self.rec.recognize_google(audio)
                        if transcript:
                            t_probs, t_top = self.text_eng.predict(transcript)
                except: pass

                # 6. Update State
                with self.lock:
                    if v_probs: self.data['voice'] = {'probs': v_probs, 'top': v_top}
                    if t_probs: self.data['text'] = {'probs': t_probs, 'top': t_top, 'content': transcript}
                
                # Cleanup
                os.remove(tmp.name)
                
            except Exception as e:
                print(f"Audio Error: {e}")

    def get_fusion(self):
        with self.lock:
            # Copy current state safely
            d = self.data.copy()
        
        final = {l: 0.0 for l in LABELS}
        active_w = 0.0
        
        # Weighted Average Logic (Only count active sensors)
        
        # Face (Always contributes if available)
        if d['face']['probs']:
            for l, s in d['face']['probs'].items(): final[l] += s * WEIGHTS['face']
            active_w += WEIGHTS['face']
            
        # Voice (Only if not silent)
        if d['voice']['probs']:
            for l, s in d['voice']['probs'].items(): final[l] += s * WEIGHTS['voice']
            active_w += WEIGHTS['voice']
            
        # Text (Only if transcript exists)
        if d['text']['probs']:
            for l, s in d['text']['probs'].items(): final[l] += s * WEIGHTS['text']
            active_w += WEIGHTS['text']
            
        # Normalize
        if active_w > 0:
            for l in final: final[l] /= active_w
            dom = max(final, key=final.get)
            conf = final[dom]
        else:
            dom = "neutral"
            conf = 0.0
            
        return dom, conf, d

# ==========================================
# 4. VISUALIZATION UTILS
# ==========================================
def draw_dashboard(img, dom, conf, data):
    h, w, _ = img.shape
    
    # Bottom Panel Background
    panel_h = 180
    cv2.rectangle(img, (0, h-panel_h), (w, h), (30, 30, 30), -1)
    
    # 1. Final Result (Left Big Box)
    color = COLORS.get(dom, (255,255,255))
    cv2.putText(img, "FINAL FUSION", (20, h-panel_h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(img, dom.upper(), (20, h-panel_h+80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(img, f"Conf: {conf:.2f}", (20, h-panel_h+120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

    # Vertical Separator
    cv2.line(img, (280, h-panel_h+10), (280, h-10), (100,100,100), 1)
    
    # 2. Individual Modalities (Right Columns)
    # Calculate column width
    start_x = 300
    col_w = (w - start_x) // 3
    headers = ["FACE", "VOICE", "TEXT"]
    keys = ['face', 'voice', 'text']
    
    for i, (head, key) in enumerate(zip(headers, keys)):
        x = start_x + i * col_w
        y = h - panel_h + 30
        
        # Title
        cv2.putText(img, head, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        
        # Value
        info = data[key]
        val = info['top'].upper()
        val_col = COLORS.get(info['top'], (200,200,200))
        
        cv2.putText(img, val, (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, val_col, 2)
        
        # Extra Info (Transcript for text)
        if key == 'text' and info.get('content'):
            short_text = (info['content'][:12] + '..') if len(info['content']) > 12 else info['content']
            cv2.putText(img, f"'{short_text}'", (x, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    # 1. Init Logic
    fusion = EmotionFusion()
    
    # 2. Start Audio Thread
    t = threading.Thread(target=fusion.process_audio, daemon=True)
    t.start()
    
    # 3. Start Camera
    cap = cv2.VideoCapture(0)
    
    print("✅ System Ready. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- Face Analysis (DeepFace) ---
        try:
            # We use silent=True and enforce_detection=False to prevent crashing
            objs = DeepFace.analyze(frame, actions=['emotion'], 
                                  detector_backend='opencv', 
                                  enforce_detection=False, 
                                  silent=True)
            
            if objs:
                # Get main face (largest)
                main_face = objs[0]
                emo_dict = main_face['emotion'] # e.g., {'angry': 0.2, ...}
                
                # Normalize DeepFace (0-100) to (0-1)
                norm_probs = {k: v/100.0 for k,v in emo_dict.items()}
                
                # Map keys to standard LABELS
                std_probs = {l: 0.0 for l in LABELS}
                for k, v in norm_probs.items():
                    if k == 'happiness': k = 'happy'
                    if k == 'sadness': k = 'sad'
                    if k in LABELS:
                        std_probs[k] = v
                        
                fusion.update_face(std_probs)
                
                # Draw Face Box
                region = main_face['region']
                x,y,w,h = region['x'], region['y'], region['w'], region['h']
                dom = fusion.data['face']['top']
                col = COLORS.get(dom, (255,255,255))
                cv2.rectangle(frame, (x,y), (x+w, y+h), col, 2)
                
        except Exception:
            pass # No face found this frame

        # --- Fusion & UI ---
        final_dom, final_conf, all_data = fusion.get_fusion()
        
        draw_dashboard(frame, final_dom, final_conf, all_data)
        
        cv2.imshow("Multimodal Emotion Fusion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()