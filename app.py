import os
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy.editor import VideoFileClip  # تم تصحيح الخطأ الإملائي هنا
import torch

app = Flask(__name__)
CORS(app)

# التحقق من توفر GPU لتحديد النموذج الأنسب
device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"  # 'base' هو توازن جيد بين الدقة والسرعة ويعمل جيدًا على CPU

print(f"Using device: {device}. Loading model: {model_size}")

# تحميل نموذج Whisper (سيتم تنزيله تلقائيًا في أول مرة)
model = whisper.load_model(model_size, device=device)
print("AI Model (Whisper) loaded successfully.")

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "Video file not found in the request."}), 400
    
    video_file = request.files['video']
    
    # إنشاء مجلد مؤقت إذا لم يكن موجودًا
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    temp_video_path = os.path.join("temp", "temp_video.mp4")
    video_file.save(temp_video_path)
    
    try:
        print("Extracting audio from video...")
        video_clip = VideoFileClip(temp_video_path)
        temp_audio_path = os.path.join("temp", "temp_audio.wav")
        video_clip.audio.write_audiofile(temp_audio_path)
        video_clip.close()
        print("Audio extracted successfully.")

        print("Starting transcription... This may take some time depending on video length.")
        # استخدام Whisper لتحويل الصوت إلى نص
        # fp16=False ضروري عند استخدام CPU
        result = model.transcribe(temp_audio_path, fp16=(device == 'cuda'))
        print("Transcription complete.")
        
        # حذف الملفات المؤقتة
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
        
        # إعادة النص الناتج
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        # التأكد من حذف الملفات المؤقتة حتى في حالة حدوث خطأ
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Video Transcription Server is ready!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
