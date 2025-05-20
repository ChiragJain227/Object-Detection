import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import cv2
from torchvision.transforms import functional as F

class FRCNN:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, min_size=800, max_size=1333)
        self.model.eval()
        self.device = torch.device('cpu')  # CPU-only as per torch==2.6.0+cpu
        self.model.to(self.device)
        self.class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                            'hair drier', 'toothbrush']

    def preprocess(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).to(self.device)
        return image_tensor

    def detect(self, image, conf_threshold=0.7, nms_threshold=0.3):
        if image is None or image.size == 0:
            return np.array([])
        
        inputs = self.preprocess(image)
        # Test-time augmentation: original + horizontal flip
        images = [inputs, F.hflip(inputs)]
        all_boxes, all_scores, all_labels = [], [], []

        with torch.no_grad():
            for img in images:
                outputs = self.model([img])
                boxes = outputs[0]['boxes'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()
                labels = outputs[0]['labels'].cpu().numpy()
                mask = scores >= conf_threshold
                boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
                if len(boxes) > 0:
                    keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), nms_threshold)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        # Merge detections
        boxes = np.concatenate(all_boxes) if all_boxes[0].size > 0 else np.array([])
        scores = np.concatenate(all_scores) if all_scores[0].size > 0 else np.array([])
        labels = np.concatenate(all_labels) if all_labels[0].size > 0 else np.array([])
        
        if len(boxes) > 0:
            keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), nms_threshold)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        
        detections = np.zeros((len(boxes), 6))
        detections[:, :4] = boxes
        detections[:, 4] = scores
        detections[:, 5] = labels
        return detections

    def draw_boxes(self, image, detections):
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls = det
            label = f"{self.class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image