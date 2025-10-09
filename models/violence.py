from transformers import ViTForImageClassification, ViTFeatureExtractor
# Violence detection model
violence_model = ViTForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection")
violence_extractor = ViTFeatureExtractor.from_pretrained("jaranohaal/vit-base-violence-detection")