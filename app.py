from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from services.text_service import split_into_clauses, analyze_clauses, generate_summary
from services.image_service import is_image_nsfw, is_image_violent, predict_image
from models.toxic import toxic_pipeline

import base64, tempfile, os

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)


# API: Sentiment Analysis
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
    
# API: Review Toxic + Image check
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

# API: Food classifier
@app.route('/callModel', methods=['POST'])
@cross_origin(supports_credentials=True)
def call_model():
    if 'file' not in request.files:
        return jsonify({"message": "Failed", "reason": "Invalid json"}), 400
    file = request.files['file']
    predicted_class, confidence = predict_image(file)
    return jsonify({"message": "Success", "productName": predicted_class, "accuracy": float(confidence)}), 200
