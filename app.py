import os
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import torch

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"

print(f"Using device: {device}. Loading model: {model_size}")
model = whisper.load_model(model_size, device=device)
print("AI Model (Whisper) loaded successfully.")

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "Video file not found."}), 400
        
    # ---===[[  التعديل الجديد: استقبال اللغة من الطلب  ]]===---
    lang = request.form.get('language', 'auto') # القيمة الافتراضية هي 'auto' للكشف التلقائي
    print(f"Received request for language: {lang}")
    # ---===[[  نهاية التعديل  ]]===---

    video_file = request.files['video']
        
    if not os.path.exists("temp"):
        os.makedirs("temp")
            
    temp_video_path = os.path.join("temp", "temp_video.mp4")
    video_file.save(temp_video_path)
        
    try:
        video_clip = VideoFileClip(temp_video_path)
        temp_audio_path = os.path.join("temp", "temp_audio.wav")
        video_clip.audio.write_audiofile(temp_audio_path)
        video_clip.close()

        # ---===[[  التعديل الجديد: استخدام متغير اللغة في Whisper  ]]===---
        if lang == 'auto':
            # إذا كانت اللغة 'auto'، لا نمرر المعامل ليقوم Whisper بالكشف التلقائي
            result = model.transcribe(temp_audio_path, fp16=(device == 'cuda'))
        else:
            # إذا تم تحديد لغة، نمررها إلى Whisper
            result = model.transcribe(temp_audio_path, language=lang, fp16=(device == 'cuda'))
        # ---===[[  نهاية التعديل  ]]===---
            
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
            
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        # ... (باقي الكود يبقى كما هو) ...
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Multi-Language Transcription Server is ready!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

