import torch

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = box1[0] + box1[2]
    y1_max = box1[1] + box1[3]
    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = box2[0] + box2[2]
    y2_max = box2[1] + box2[3]
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

def compute_metrics(outputs, targets, iou_threshold=0.5):
    """Compute IoU, accuracy, precision, and recall for a batch."""
    batch_size = len(outputs['label'])
    correct = 0
    total = batch_size
    tp = 0
    fp = 0
    fn = 0
    iou_list = []
    for i in range(batch_size):
        pred_bbox = outputs['bbox'][i]
        pred_label = outputs['label'][i]
        true_bbox = targets['bbox'][i]
        true_label = targets['label'][i]
        iou = compute_iou(pred_bbox.cpu().numpy(), true_bbox.cpu().numpy())
        iou_list.append(iou)
        if iou >= iou_threshold:
            if pred_label == true_label:
                tp += 1
                correct += 1
            else:
                fp += 1
        else:
            fn += 1
    accuracy = correct / total
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mean_iou = sum(iou_list) / len(iou_list)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'mean_iou': mean_iou
    }
