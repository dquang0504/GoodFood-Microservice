import os.path
import tensorflow as tf

# Load model TFLite for image recognition
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