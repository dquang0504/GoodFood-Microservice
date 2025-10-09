import numpy as np
from PIL import Image
import torch

from models.nsfw import nsfw_classifier
from models.violence import violence_model, violence_extractor
from models.food_classifier import interpreter, classNames

# NSFW detection
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


# Violence detection
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
    
# Food classification
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