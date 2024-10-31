import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import numpy as np
import os
from PIL import Image
import json
from datetime import datetime
import openai

# Đường dẫn tới các file trong cùng thư mục
local_keras_model_path = 'best_model_optimized.keras'
json_path = 'advice_and_prescriptions.json'

# Load mô hình Keras từ file cục bộ
model = load_model(local_keras_model_path)

# Mở và load file JSON chứa tư vấn và đơn thuốc
with open(json_path, 'r') as json_file:
    advice_and_prescriptions = json.load(json_file)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def prepare_image(image, target_size=(150, 150)):
    if image.size != target_size:
        image = image.resize(target_size)
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def calculate_severity(prediction_probabilities):
    try:
        severity_score = np.max(prediction_probabilities)
        print(f"Calculated severity score: {severity_score}")
        return float(severity_score)  # Chuyển thành float để JSON serializable
    except Exception as e:
        print(f"Error calculating severity: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kiểm tra tham số 'files'
        if 'files' not in request.files:
            print("Error: 'files' not found in request")
            return jsonify({'error': 'No files part'}), 400

        files = request.files.getlist('files')
        print(f"Received {len(files)} files")

        if len(files) == 0:
            print("Error: No selected files")
            return jsonify({'error': 'No selected files'}), 400

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Kiểm tra tham số 'symptoms' (triệu chứng)
        symptoms = request.form.get('symptoms', '')
        if not symptoms:
            print("Error: 'symptoms' not found")
            return jsonify({'error': 'No symptoms provided'}), 400

        all_predictions = []
        saved_files = []
        severity_scores = []
        predicted_diseases = []

        for file in files:
            try:
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                saved_files.append(file_path)
                print(f"File saved to {file_path}")

                # Đọc ảnh gốc và không áp dụng YOLO hay augmentation
                img = load_img(file_path, target_size=(150, 150))
                img = prepare_image(img)
                prediction = model.predict(img)[0]  # Lấy dự đoán của mô hình

                # In tỉ lệ % cho từng bệnh
                disease_names = list(advice_and_prescriptions.keys())
                print("Predicted probabilities for each disease:")
                for i, prob in enumerate(prediction):
                    print(f"{disease_names[i]}: {prob * 100:.2f}%")  # In ra xác suất dưới dạng %

                # Tính độ nghiêm trọng
                severity = calculate_severity(prediction)
                if severity is not None:
                    severity_scores.append(severity)

                # Lọc các bệnh có xác suất > threshold (ví dụ: 0.1)
                threshold = 0.1
                predicted_classes = [i for i, prob in enumerate(prediction) if prob >= threshold]
                all_predictions.extend(predicted_classes)

                # Thêm các bệnh dự đoán vào danh sách
                for prediction_class in predicted_classes:
                    predicted_diseases.append(disease_names[prediction_class])

            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")

        print(f"Predicted diseases: {predicted_diseases}")

        most_common_severity = float(np.mean(severity_scores)) if severity_scores else None

        matched_disease = None
        symptoms_lower = symptoms.lower()  # Chuyển triệu chứng thành chữ thường để so sánh không phân biệt chữ hoa/chữ thường
        for disease in predicted_diseases:
            # So sánh một phần, không phân biệt chữ hoa/chữ thường
            if disease.lower() in symptoms_lower:
                matched_disease = disease
                break

        # Luôn trả về kết quả dự đoán của mô hình (Keras)
        gpt_response = chat_with_openai(symptoms, matched_disease if matched_disease else predicted_diseases[0])
        
        response = {
            'conclusion': predicted_diseases[0] if predicted_diseases else None,  # Kết quả dự đoán từ mô hình
            'severity': most_common_severity,  # Độ nghiêm trọng
            'advice_and_prescription': gpt_response  # Phản hồi từ GPT
        }
        
        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': str(e)}), 500



# Hàm chat với GPT, cập nhật để nhận triệu chứng
def chat_with_openai(symptoms, disease=None):
    openai.api_key = 'sk-proj-1Q52RZEGbAAG2wC8wiQVT3BlbkFJMcAg0u4r8GDAuVq2FPiW'  # Hãy lưu ý bảo mật API key
    
    # Nếu có tên bệnh dự đoán từ ảnh, thêm vào yêu cầu GPT
    if disease:
        prompt = f"The patient has the following symptoms: {symptoms}. The predicted disease based on the image is {disease}. Please provide advice and a prescription."
    else:
        prompt = f"The patient has the following symptoms: {symptoms}. Please provide advice and a prescription."
    
    messages = [{"role": "system", "content": "You are an intelligent medical assistant."}]
    messages.append({"role": "user", "content": prompt})
    
    chat = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=messages
    )
    
    reply = chat.choices[0].message['content'].strip()
    print(f"ChatGPT: {reply}")
    
    return reply


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
