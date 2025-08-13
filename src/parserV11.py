import numpy as np
import onnxruntime as ort
import yaml
import matplotlib.pyplot as plt
import cv2
import time

class YOLODetector:
    def __init__(self, model_path, class_yaml, conf_thresh=0.5, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load classes from YAML
        with open(class_yaml, 'r') as f:
            self.CLASSES = yaml.safe_load(f)['names']
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        # Load model
        self.session = ort.InferenceSession(model_path)
        input_shape = self.session.get_inputs()[0].shape
        self.INPUT_H = input_shape[2]
        self.INPUT_W = input_shape[3]

    def preprocess(self, image):
        img_resized = cv2.resize(image, (self.INPUT_W, self.INPUT_H))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    def postprocess(self, outputs, orig_h, orig_w):
        preds = outputs[0]
        preds = np.transpose(preds, (0, 2, 1))[0]
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        confidences = np.max(scores_all, axis=1)

        mask = confidences > self.conf_thresh
        boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]

        boxes = []
        for cx, cy, w, h in boxes_xywh:
            xmin = max(0, (cx - w / 2) * orig_w / self.INPUT_W)
            ymin = max(0, (cy - h / 2) * orig_h / self.INPUT_H)
            xmax = min(orig_w, (cx + w / 2) * orig_w / self.INPUT_W)
            ymax = min(orig_h, (cy + h / 2) * orig_h / self.INPUT_H)
            boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        idxs = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), self.conf_thresh, self.iou_thresh)
        final_boxes, final_scores, final_class_ids = [], [], []
        if len(idxs) > 0:
            for i in idxs.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(confidences[i])
                final_class_ids.append(class_ids[i])
        return final_boxes, final_scores, final_class_ids

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        img_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_tensor})
        boxes,score,class_ids= self.postprocess(outputs,orig_h,orig_w)
        return boxes,score,class_ids

