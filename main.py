import base64
import os.path
import tempfile

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import pipeline, ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
from sentence_splitter import SentenceSplitter
from nudenet import NudeDetector

# ️️ Dùng sentence splitter đa ngôn ngữ
splitter = SentenceSplitter(language='en')  # hoặc 'en', nhưng 'vi' vẫn tách được cả 2 tốt
# 1 Toxic text detection pipeline
toxic_pipeline = pipeline(
    "text-classification",
    model="unitary/toxic-bert"  # Ví dụ model tiếng Anh chuyên detect toxicity
)
# 2 NSFW model
nsfw_classifier = NudeDetector()
# 3 Violence detection model
violence_model = ViTForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection")
violence_extractor = ViTFeatureExtractor.from_pretrained("jaranohaal/vit-base-violence-detection")
# 4 Load model TFLite for image recognition
# Danh sách class names cho Image Classifier
classNames = [
    "Bánh flan", "Bánh mì ngọt", "Bánh mochi", "Bánh tiramisu",
    "Chè thái", "Cơm bò lúc lắc", "Cơm cá chiên", "Cơm chiên dương châu", "Cơm gà", "Cơm tấm",
    "Cơm thịt kho", "Cơm xá xíu", "Kem dừa", "Kem socola", "Nước ngọt 7up", "Nước ngọt coca-cola",
    "Nước ngọt pepsi", "Nước ngọt sprite", "Nước tăng lực red bull", "Nước tăng lực sting", "Thịt bò hầm tiêu xanh",
    "Thịt heo quay"
]
model_path = os.getenv("MODEL_PATH", "model_unquant.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

app = Flask(__name__)
app.debug = True
CORS(app, origins=["*"], supports_credentials=True)

# Cấu hình cho Sentiment Analysis model
model_path = "5CD-AI/Vietnamese-Sentiment-visobert"
sent_pipeline = pipeline("text-classification", model=model_path)


# Hàm phân tách câu
def split_into_clauses(sentence):
    """
    Chia văn bản thành các cụm câu dựa trên mô hình xử lý tiếng Việt.
    """
    clauses = splitter.split(text=sentence)
    print(clauses)
    return clauses


# Hàm phân tích cảm xúc của từng câu
def analyze_clauses(clauses):
    result = []
    for clause in clauses:
        sentiment = sent_pipeline(clause)[0]
        result.append({"clause": clause, "sentiment": sentiment["label"]})
    return result


# Hàm sinh nhận xét từ phân tích cảm xúc
def generate_summary(analysis):
    summary = []
    for item in analysis:
        clause = item['clause']
        sentiment = item['sentiment']
        if sentiment == "POS":
            summary.append(f"Khen {clause.strip()}.")
        elif sentiment == "NEG":
            summary.append(f"Chê {clause.strip()}.")
        else:
            summary.append(f"Ý kiến trung lập về {clause.strip()}.")
    return " ".join(summary)


# API phân tích cảm xúc
@app.route('/analyze', methods=['POST'])
@cross_origin(supports_credentials=True)
def analyze():
    try:
        data = request.get_json()
        if not data or 'review' not in data:
            return jsonify({"error": "Json không hợp lệ"}), 400

        reviews = data['review']
        reviewIds = data['reviewID']

        # Nếu dữ liệu là một mảng chứa một mảng con, ta flatten thành một mảng chuỗi
        if isinstance(reviews[0], list):
            reviews = [item for sublist in reviews for item in sublist]

        if not isinstance(reviews, list) or not isinstance(reviewIds, list):
            return jsonify({"error": "Số lượng review và reviewID không khớp"}), 400

        results = []
        # appending review and reviewID into results
        for review, reviewID in zip(reviews, reviewIds):
            if not isinstance(review, str) or not review.strip():
                results.append({"reviewID": reviewID, "review": review, "error": "Bình luận không hợp lệ"})
                continue
            clauses = split_into_clauses(review)
            analysis = analyze_clauses(clauses)
            summary = generate_summary(analysis)
            results.append({
                "reviewID": reviewID,
                "review": review,
                "clauses": clauses,
                "analysis": analysis,
                "summary": summary
            })
        print(results)
        return jsonify(results), 200  # Trả về tất cả kết quả cho tất cả bình luận
    except Exception as e:
        print(str(e))
        return jsonify({"error": "Đã xảy ra lỗi khi xử lý", "details": str(e)}), 500


# Hàm kiểm tra NSFW
NSFW_CLASSES = {
    "GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BELLY_EXPOSED",  # optional: nhẹ
}
NSFW_THRESHOLD = 0.6  # ngưỡng confidence


def is_image_nsfw(image_path):
    results = nsfw_classifier.detect(image_path)

    if isinstance(results, list) and len(results) > 0:
        scores = results[0]
    else:
        return False, {}

    detected_class = scores.get("class", "")
    confidence = scores.get("score", 0)

    is_nsfw = detected_class in NSFW_CLASSES and confidence >= NSFW_THRESHOLD

    return is_nsfw, scores


# Hàm kiểm tra bạo lực
def is_image_violent(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")

    try:
        # Tiền xử lý ảnh
        inputs = violence_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = violence_model(**inputs)

        # Tính xác suất
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        score = probs[pred_idx].item() * 100

        # Nhãn gốc như Space
        violence_labels = ["Violent", "Non-Violent"]
        violence_label = violence_labels[pred_idx]  # Chuẩn giống Space

        # In debug
        print(f"[Violence] label={violence_label}, score={score:.2f}%")
        print(f"[Violence] probs={probs.tolist()}")

        # Áp dụng logic y chang Hugging Face Space
        is_violent = False
        if violence_label.lower() == "non-violent" and score > 65:
            is_violent = True
        elif violence_label.lower() == "violent" and score > 80:
            is_violent = True

        return is_violent, violence_label, score

    except Exception as e:
        raise ValueError(f"Violence model failed: {e}")


# Hàm xử lý lọc bình luận
@app.route('/reviewLabel', methods=['POST'])
def detect_review():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()
        review = data.get("review", "")
        image_dict = data.get("images", {})

        # checking for toxic text
        if not review:
            return jsonify({"error": "No review provided"}), 400
        print(review)
        # Xử lý dự đoán
        prediction = toxic_pipeline(review)[0]
        print(prediction)

        # Threshold
        threshold = 0.5
        is_toxic = prediction["score"] >= threshold
        label = "toxic" if is_toxic else "non_toxic"

        # Checking every image
        image_results = []
        for filename, image_bytes in image_dict.items():
            try:
                # 1. Chuyển từ base64 string nếu cần, hoặc raw bytes
                if isinstance(image_bytes, str):
                    image_data = base64.b64decode(image_bytes)
                else:
                    image_data = bytes(image_bytes)  # fallback raw bytes

                # 2. Lưu tạm vào file để dùng trong các model hiện tại
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name

                # 3. Phân tích ảnh
                nsfw_flag, nsfw_score = is_image_nsfw(tmp_path)
                violent_flag, violent_label, violent_score = is_image_violent(tmp_path)

                print(violent_flag)
                print(violent_label)

                image_results.append({
                    "image": filename,
                    "nsfw": nsfw_flag,
                    "nsfw_scores": nsfw_score,
                    "violent": violent_flag,
                    "violent_label": violent_label,
                    "violent_score": violent_score
                })

                # Xoá file tạm
                os.remove(tmp_path)

            except Exception as e:
                print(f"Failed to process image {filename}: {e}")
                image_results.append({
                    "image": filename,
                    "error": str(e)
                })

        return jsonify({"label": label, "score": prediction["score"], "images": image_results}), 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "Đã xảy ra lỗi", "details": str(e)}), 500


# Hàm xử lý ảnh
def preprocess_image(file, target_size):
    image = Image.open(file).resize(target_size)
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Chuẩn hóa về [0, 1]
    image = np.expand_dims(image, axis=0)
    return image


# Dự đoán lớp của ảnh
def predict_image(file):
    input_shape = interpreter.get_input_details()[0]['shape'][1:3]
    input_data = preprocess_image(file, input_shape)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    predicted_index = np.argmax(output_data)
    predicted_class = classNames[predicted_index]
    return predicted_class, output_data[0][predicted_index]


# API nhận diện ảnh
@app.route('/callModel', methods=['POST'])
@cross_origin(supports_credentials=True)
def call_model():
    if 'file' not in request.files:
        return jsonify({"message": "Failed", "reason": "Invalid json"}), 400
    file = request.files['file']
    predicted_class, confidence = predict_image(file)
    return jsonify({"message": "Success", "productName": predicted_class, "accuracy": float(confidence)}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
