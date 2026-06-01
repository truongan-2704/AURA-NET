"""
Bounding box utilities
"""

import torch
import torchvision


def xywh2xyxy(boxes):
    """
    Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2)
    
    Args:
        boxes: Tensor (..., 4)
    
    Returns:
        boxes: Tensor (..., 4)
    """
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy2xywh(boxes):
    """
    Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, w, h)
    
    Args:
        boxes: Tensor (..., 4)
    
    Returns:
        boxes: Tensor (..., 4)
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x, y, w, h], dim=-1)


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        box1: Tensor (N, 4) in format (x1, y1, x2, y2)
        box2: Tensor (M, 4) in format (x1, y1, x2, y2)
    
    Returns:
        iou: Tensor (N, M)
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def non_max_suppression(detections, iou_thres=0.45, max_det=300):
    """
    Apply Non-Maximum Suppression
    
    Args:
        detections: List of tensors, each (N, 9) containing
                   [x1, y1, x2, y2, obj_conf, class_conf, class_id, evidence, uncertainty]
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections to keep
    
    Returns:
        output: List of tensors after NMS
    """
    output = []
    
    for det in detections:
        if len(det) == 0:
            output.append(det)
            continue
        
        # Get boxes and scores
        boxes = det[:, :4]
        scores = det[:, 4] * det[:, 5]  # obj_conf * class_conf
        classes = det[:, 6]
        
        # NMS per class
        keep = []
        unique_classes = classes.unique()
        
        for c in unique_classes:
            mask = classes == c
            boxes_c = boxes[mask]
            scores_c = scores[mask]
            indices_c = torch.where(mask)[0]
            
            # Apply NMS using torchvision
            keep_c = torchvision.ops.nms(boxes_c, scores_c, iou_thres)
            keep.append(indices_c[keep_c])
        
        if len(keep) > 0:
            keep = torch.cat(keep)
            det = det[keep]
            
            # Limit to max_det
            if len(det) > max_det:
                det = det[:max_det]
        
        output.append(det)
    
    return output


def clip_boxes(boxes, img_shape):
    """
    Clip boxes to image boundaries
    
    Args:
        boxes: Tensor (..., 4) in format (x1, y1, x2, y2)
        img_shape: (height, width)
    
    Returns:
        boxes: Clipped boxes
    """
    boxes[..., 0] = boxes[..., 0].clamp(0, img_shape[1])
    boxes[..., 1] = boxes[..., 1].clamp(0, img_shape[0])
    boxes[..., 2] = boxes[..., 2].clamp(0, img_shape[1])
    boxes[..., 3] = boxes[..., 3].clamp(0, img_shape[0])
    return boxes


def scale_boxes(boxes, from_shape, to_shape):
    """
    Scale boxes from one image shape to another
    
    Args:
        boxes: Tensor (..., 4) in format (x1, y1, x2, y2)
        from_shape: (height, width) of source image
        to_shape: (height, width) of target image
    
    Returns:
        boxes: Scaled boxes
    """
    gain = min(to_shape[0] / from_shape[0], to_shape[1] / from_shape[1])
    boxes = boxes * gain
    return boxes
