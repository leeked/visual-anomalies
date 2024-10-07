import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def get_map_metric(iou_thresholds=None, class_metrics=False):
    # iou_thresholds can be a list or a single float
    # if iou_thresholds is None:
    #     iou_thresholds = [0.5]
    return MeanAveragePrecision(iou_thresholds=iou_thresholds, class_metrics=class_metrics)

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    # box1: [xmin, ymin, xmax, ymax]
    # box2: [xmin, ymin, xmax, ymax]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def match_predictions_to_ground_truth(pred_boxes, pred_labels, pred_scores,
                                      gt_boxes, gt_labels, iou_threshold=0.5,
                                      index_to_class_num=None):
    """
    Matches predicted boxes to ground truth boxes.
    Returns counts of TP, FP, FN, and a list of IoUs for matched pairs.
    """
    pred_boxes = pred_boxes.numpy()
    pred_labels = pred_labels.numpy()
    pred_scores = pred_scores.numpy()
    gt_boxes = gt_boxes.numpy()
    gt_labels = gt_labels.numpy()

    if index_to_class_num is not None:
        # Map indices back to original class numbers
        pred_labels = np.array([index_to_class_num[int(label)] for label in pred_labels])
        gt_labels = np.array([index_to_class_num[int(label)] for label in gt_labels])

    matched_gt = []
    tp = 0
    fp = 0
    iou_list = []

    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = pred_labels[i]
        best_iou = 0.0
        best_j = -1
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            gt_box = gt_boxes[j]
            gt_label = gt_labels[j]
            if pred_label != gt_label:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.append(best_j)
            iou_list.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn, iou_list

def compute_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1_score

def compute_mean_iou(iou_list):
    if iou_list:
        return sum(iou_list) / len(iou_list)
    else:
        return 0.0
