import cv2
import threading
import time
import numpy as np
from fusion_module import EmotionFusion, TextEmotionAnalyzer, VoiceEmotionAnalyzer, draw_dashboard, LABELS, COLORS
from deepface import DeepFace
import database

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        if not self.video.isOpened():
            print("Cannot open camera")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        with self.lock:
            success, image = self.video.read()
        if not success:
            return None
        return image

class EmotionService:
    def __init__(self):
        self.fusion = EmotionFusion()
        self.camera = VideoCamera()
        self.mode = 'fusion' # 'fusion', 'face', 'raw'
        self.running = True
        
        # Start audio thread for fusion
        self.audio_thread = threading.Thread(target=self.fusion.process_audio, daemon=True)
        self.audio_thread.start()

    def get_video_feed(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Process frame based on mode
            if self.mode == 'face' or self.mode == 'fusion':
                try:
                    # Face Detection
                    objs = DeepFace.analyze(frame, actions=['emotion'], 
                                          detector_backend='opencv', 
                                          enforce_detection=False, 
                                          silent=True)
                    
                    if objs:
                        main_face = objs[0]
                        emo_dict = main_face['emotion']
                        
                        # Normalize and Update Fusion
                        norm_probs = {k: v/100.0 for k,v in emo_dict.items()}
                        std_probs = {l: 0.0 for l in LABELS}
                        for k, v in norm_probs.items():
                            if k == 'happiness': k = 'happy'
                            if k == 'sadness': k = 'sad'
                            if k in LABELS:
                                std_probs[k] = v
                        
                        self.fusion.update_face(std_probs)
                        
                        # Draw Box
                        region = main_face['region']
                        x,y,w,h = region['x'], region['y'], region['w'], region['h']
                        
                        # Determine label to show
                        if self.mode == 'face':
                            # For face mode, show just the face emotion
                            top_face = max(std_probs, key=std_probs.get)
                            col = COLORS.get(top_face, (255,255,255))
                            cv2.rectangle(frame, (x,y), (x+w, y+h), col, 2)
                            cv2.putText(frame, f"{top_face}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                            
                            # Log occasionally? (Maybe too frequent)
                        else:
                            # Fusion mode
                            dom = self.fusion.data['face']['top']
                            col = COLORS.get(dom, (255,255,255))
                            cv2.rectangle(frame, (x,y), (x+w, y+h), col, 2)

                except Exception as e:
                    pass

            if self.mode == 'fusion':
                final_dom, final_conf, all_data = self.fusion.get_fusion()
                draw_dashboard(frame, final_dom, final_conf, all_data)
                
                # Log significant changes or periodically could be done here
                # For now, we rely on explicit user actions for logging in other modes, 
                # but fusion is continuous.

            # Encode frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    def process_text(self, text):
        probs, top = self.fusion.text_eng.predict(text)
        if top:
            conf = probs[top]
            database.log_emotion("text", text, top, conf)
            return {"emotion": top, "confidence": conf, "probs": probs}
        return {"error": "Could not process text"}

    def process_voice_file(self, filepath):
        probs, top = self.fusion.voice_eng.predict(filepath)
        if top:
            conf = probs[top]
            database.log_emotion("voice", "uploaded_file", top, conf)
            return {"emotion": top, "confidence": conf, "probs": probs}
        return {"error": "Could not process audio"}

# Singleton instance
service = EmotionService()
