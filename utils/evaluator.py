import torch
from utils.metrics import (
    compute_precision_recall_f1, compute_mean_iou,
    match_predictions_to_ground_truth, get_map_metric
)


class Evaluator:
    def __init__(self, model, device, dataloader, dataset, config):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.dataset = dataset
        self.config = config

        self.iou_thresholds = config['metrics'].get(
            'iou_thresholds', [0.5]
        )
        self.matching_iou_threshold = config['metrics'].get(
            'matching_iou_threshold', 0.5
        )

        self.metric = get_map_metric(
            iou_thresholds=self.iou_thresholds, class_metrics=True
        )

        self.index_to_class_num = dataset.index_to_class_num

    def evaluate(self):
        self.model.eval()
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou_list = []

        with torch.no_grad():
            for images, targets in self.dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                output = outputs[0]
                target = targets[0]

                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()

                gt_boxes = target['boxes']
                gt_labels = target['labels']

                # Map predicted labels back to original class numbers
                pred_labels_orig = torch.tensor([
                    self.index_to_class_num[int(label)]
                    for label in pred_labels
                ])

                gt_labels_orig = torch.tensor([
                    self.index_to_class_num[int(label)]
                    for label in gt_labels
                ])

                preds = [{
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                }]

                target_formatted = [{
                    'boxes': gt_boxes,
                    'labels': gt_labels
                }]

                self.metric.update(preds, target_formatted)

                tp, fp, fn, iou_list = match_predictions_to_ground_truth(
                    pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels,
                    iou_threshold=self.matching_iou_threshold,
                    index_to_class_num=self.index_to_class_num
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_iou_list.extend(iou_list)

            self._print_results(total_tp, total_fp, total_fn, total_iou_list)

    def _print_results(self, total_tp, total_fp, total_fn, total_iou_list):
        final_metrics = self.metric.compute()
        print('Evaluation Results:')
        for k, v in final_metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                    print(f'{k}: {v:.4f}')
                else:
                    v_list = v.tolist()
                    v_str = ', '.join(f'{val:.4f}' for val in v_list)
                    print(f'{k}: [{v_str}]')
            else:
                print(f'{k}: {v}')

        precision, recall, f1_score = compute_precision_recall_f1(
            total_tp, total_fp, total_fn
        )
        mean_iou = compute_mean_iou(total_iou_list)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1_score:.4f}')
        print(f'Mean IoU: {mean_iou:.4f}')
