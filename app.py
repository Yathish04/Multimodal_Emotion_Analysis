from flask import Flask, render_template, Response, request, jsonify
import os
import database
from emotion_service import service

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize DB
database.init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'fusion')
    service.mode = mode
    
    # Pause audio loop if not in fusion mode to free up resources/mic
    if mode == 'fusion':
        service.fusion.paused = False
    else:
        service.fusion.paused = True
        
    return Response(service.get_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/text_emotion', methods=['POST'])
def text_emotion():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = service.process_text(text)
    return jsonify(result)

@app.route('/api/voice_emotion', methods=['POST'])
def voice_emotion():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Secure filename and save
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    print(f"Processing audio file: {filepath}")
    result = service.process_voice_file(filepath)
    
    # Cleanup
    try:
        os.remove(filepath)
    except:
        pass
        
    return jsonify(result)

@app.route('/api/logs')
def get_logs():
    logs = database.get_logs()
    return jsonify(logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
