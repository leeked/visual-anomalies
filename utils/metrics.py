import torch

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    # box1: [xmin, ymin, xmax, ymax]
    # box2: [xmin, ymin, xmax, ymax]
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = box1[2]
    y1_max = box1[3]
    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = box2[2]
    y2_max = box2[3]
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

def compute_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """Compute IoU, accuracy, precision, and recall for a single image."""
    # pred_boxes: [N_pred, 4]
    # pred_labels: [N_pred]
    # gt_boxes: [N_gt, 4]
    # gt_labels: [N_gt]

    pred_boxes = pred_boxes.numpy()
    pred_labels = pred_labels.numpy()
    gt_boxes = gt_boxes.numpy()
    gt_labels = gt_labels.numpy()

    num_gt = len(gt_labels)
    num_pred = len(pred_labels)

    matched = []
    tp = 0
    fp = 0
    fn = 0
    iou_list = []

    for i in range(num_pred):
        pred_box = pred_boxes[i]
        pred_label = pred_labels[i]
        best_iou = 0.0
        best_j = -1
        for j in range(num_gt):
            if j in matched:
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
            matched.append(best_j)
            iou_list.append(best_iou)
        else:
            fp += 1

    fn = num_gt - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    accuracy = tp / num_gt if num_gt > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'mean_iou': mean_iou
    }
