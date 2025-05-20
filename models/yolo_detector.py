from ultralytics import YOLO as UltralyticsYOLO
import cv2
import numpy as np

class YoloDetector:
    def __init__(self, model_variant="yolov5l"):  # Switched to yolov5l for higher accuracy
        try:
            self.model = UltralyticsYOLO(f"{model_variant}.pt")
            self.class_names = self.model.names
            print(f"YOLOv5 {model_variant} model loaded successfully")
            print(f"Class names: {self.class_names}")
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            raise

    def detect(self, image, conf_threshold=0.4, iou_threshold=0.3):
        if image is None or image.size == 0:
            print("Invalid image input")
            return np.array([])
        
        # Run inference with higher resolution and test-time augmentation
        results = self.model(image, conf=conf_threshold, iou=iou_threshold, imgsz=1280, augment=True)
        detections = results[0].boxes.data.cpu().numpy()
        print(f"Number of detections: {len(detections)}")
        if len(detections) > 0:
            for i, det in enumerate(detections):
                x_min, y_min, x_max, y_max, conf, cls = det
                class_name = self.class_names[int(cls)]
                print(f"Detection {i}: Class={class_name}, Conf={conf:.2f}, Box=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
        return detections

    def draw_boxes(self, image, detections):
        print(f"Drawing {len(detections)} boxes")
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls = det
            class_name = self.class_names[int(cls)]
            label = f"{class_name}: {conf:.2f}"
            print(f"Drawing: {label}, Box=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

if __name__ == "__main__":
    detector = YoloDetector()
    img = cv2.imread("test.jpg")
    detections = detector.detect(img)
    result = detector.draw_boxes(img, detections)
    cv2.imwrite("output.jpg", result)