# needed for model
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

BASE = 40
WIDTH = 2 * 40
HEIGHT = 2 * 40

NUM_CLASSES = 10

S = 7
B = 2
l_coord = 5
l_noobj = 0.5

cuda = torch.cuda.is_available()
#cuda = False

device = 'cuda:0' if cuda else 'cpu'

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if cuda else torch.BoolTensor


def encode_target(boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
            labels: (tensor) [c_obj1, c_obj2, ...]
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """
        C = NUM_CLASSES
        N = 5 * B + C

        target = torch.zeros(S, S, N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]
        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size # x & y of the box on the cell, normalized from 0.0 to 1.0.

            for k in range(B):
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0
            target[j, i, 5*B + label] = 1.0

        return target
    
def decode_pred(pred_tensor, conf_thresh=0.1, prob_thresh=0.1):
        """ Decode tensor into box coordinates, class labels, and probs_detected.
        Args:
            pred_tensor: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        Returns:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
            labels: (tensor) class labels for each detected boxe, sized [n_boxes,].
            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
            class_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
        """
        C = NUM_CLASSES
        boxes, labels, confidences, class_scores = [], [], [], []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2) # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        conf_mask = conf > conf_thresh # [S, S, B]

        # TBM, further optimization may be possible by replacing the following for-loops with tensor operations.
        for i in range(S): # for x-dimension.
            for j in range(S): # for y-dimension.
                class_score, class_label = torch.max(pred_tensor[j, i, 5*B:], 0)

                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score
                    if float(prob) < prob_thresh:
                        continue

                    # Compute box corner (x1, y1, x2, y2) from tensor.
                    box = pred_tensor[j, i, 5*b : 5*b + 4]
                    x0y0_normalized = FloatTensor([i, j]) * cell_size # cell left-top corner. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    xy_normalized = box[:2] * cell_size + x0y0_normalized   # box center. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    wh_normalized = box[2:] # Box width and height. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    box_xyxy = FloatTensor(4) # [4,]
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized # left-top corner (x1, y1).
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized # right-bottom corner (x2, y2).

                    # Append result to the lists.
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0) # [n_boxes, 4]
            labels = torch.stack(labels, 0)             # [n_boxes, ]
            confidences = torch.stack(confidences, 0)   # [n_boxes, ]
            class_scores = torch.stack(class_scores, 0) # [n_boxes, ]
        else:
            # If no box found, return empty tensors.
            boxes = FloatTensor(0, 4)
            labels = LongTensor(0)
            confidences = FloatTensor(0)
            class_scores = FloatTensor(0)

        return boxes, labels, confidences, class_scores


def nms(boxes, scores, nms_thresh = 0.25):
    """ Apply non maximum supression.
    Args:
    Returns:
    """
    threshold = nms_thresh

    x1 = boxes[:, 0] # [n,]
    y1 = boxes[:, 1] # [n,]
    x2 = boxes[:, 2] # [n,]
    y2 = boxes[:, 3] # [n,]
    areas = (x2 - x1) * (y2 - y1) # [n,]

    _, ids_sorted = scores.sort(0, descending=True) # [n,]
    ids = []
    while ids_sorted.numel() > 0:
        # Assume `ids_sorted` size is [m,] in the beginning of this iter.

        i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        ids.append(i)

        if ids_sorted.numel() == 1:
            break # If only one box is left (i.e., no box to supress), break.

        inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i].item()) # [m-1, ]
        inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i].item()) # [m-1, ]
        inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i].item()) # [m-1, ]
        inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i].item()) # [m-1, ]
        inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

        inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
        ious = inters / unions # [m-1, ]

        # Remove boxes whose IoU is higher than the threshold.
        ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        if ids_keep.numel() == 0:
            break # If no box left, break.
        ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

    return LongTensor(ids)


def process_target(target):
    
    out_target = []
    for idx in range(len(target)):
        
        #how many boxes for these target
        nbox = target[idx]['bounding_box'].shape[0]
        individual_target = FloatTensor(nbox, 14).fill_(0)
        
        bbox = target[idx]['bounding_box'].to(device)
        translation = FloatTensor(bbox.shape[0], bbox.shape[1], bbox.shape[2])
        translation[:, 0, :].fill_(-40)
        translation[:, 1, :].fill_(40)

        # translate to uppert left
        box = bbox - translation
        # reflect y
        box[:, 1, :].mul_(-1)

        x_min = box[:, 0].min(dim = 1)[0]
        y_min = box[:, 1].min(dim = 1)[0]
        x_max = box[:, 0].max(dim = 1)[0]
        y_max = box[:, 1].max(dim = 1)[0]

        x_min = x_min / WIDTH
        y_min = y_min / HEIGHT
        x_max = x_max / WIDTH
        y_max = y_max / HEIGHT
        
        boxes = torch.stack([x_min, y_min, x_max, y_max], 1)

        labels = IntTensor(nbox)
        for box_index in range(nbox):
            category = target[idx]['category'][box_index]
            labels[box_index] = category
            
        individual_target = encode_target(boxes, labels)
        out_target.append(individual_target)
        
    return torch.stack(out_target, dim = 0)

ENCODER_HIDDEN = int(26718 / 2)

class YoloEncoder(nn.Module):
    def __init__(self, n_features):
        super(YoloEncoder, self).__init__()
        # number of different kernels to use
        self.n_features = n_features
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=n_features,
                               kernel_size=5,
                               )
        self.conv2 = nn.Conv2d(n_features,
                               int(n_features / 2),
                               kernel_size=5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # return an array shape
        x = x.view(-1, ENCODER_HIDDEN)
        return x


class ReshapeLayer2d(nn.Module):
    def __init__(self, channels, dim):
        super(ReshapeLayer2d, self).__init__()
        self.channels = channels
        self.dim = dim

    def forward(self, x):
        return x.view(x.shape[0], self.channels, self.dim, self.dim)


class ReshapeLayer1d(nn.Module):
    def __init__(self, features):
        super(ReshapeLayer1d, self).__init__()
        self.features = features

    def forward(self, x):
        return x.view(x.shape[0], self.features)


class YoloDecoder(nn.Module):
    def __init__(self, num_classes):
        
        super(YoloDecoder, self).__init__()
        self.num_classes = num_classes
        # takes in dense output from encoder or shared decoder and puts into an
        # image of dim img_dim

        self.m = nn.Sequential(
            nn.Linear(6 * ENCODER_HIDDEN, 2 * 15 * 15),
            nn.ReLU(),
            ReshapeLayer2d(2, 15),
            nn.Conv2d(2, 2, kernel_size=3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 1),
            ReshapeLayer1d(288),
            nn.Linear(288, S * S * (5 * B + self.num_classes)),
            # Sigmoid is final layer in Yolo v1
            nn.Sigmoid()
        )
        
    def forward(self, x):

        x = self.m(x)
        num_samples = x.shape[0]
        prediction = (
            x.view(num_samples, S, S, 5 * B + self.num_classes)
            .contiguous()
        )
        return prediction


# from https://github.com/motokimura/yolo_v1_pytorch/
class YoloLoss(nn.Module):
    def __init__(self, feature_size=S, num_bboxes=B, num_classes=NUM_CLASSES, 
                 lambda_coord=l_coord, lambda_noobj=l_noobj):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(YoloLoss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.

        S, B, C = self.S, self.B, self.C
        
        N = 5 * B + C

        batch_size = pred_tensor.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0 # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred_tensor[coord_mask].view(-1, N)            # pred tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)    # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5*B:]                            # [n_coord, C]

        coord_target = target_tensor[coord_mask].view(-1, N)        # target tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)# [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5*B:]                        # [n_coord, C]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)        # pred tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1, N)    # target tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_conf_mask = BoolTensor(noobj_pred.size()).fill_(0) # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b*5] = 1 # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]       # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]   # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = BoolTensor(bbox_target.size()).fill_(0)    # [n_coord x B, 5]
        coord_not_response_mask = BoolTensor(bbox_target.size()).fill_(1)# [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size())                    # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size())) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:,  :2] = pred[:, 2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, 2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size())) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, 2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, 2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4]) # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, LongTensor([4])] = (max_iou).data
        bbox_target_iou = Variable(bbox_target_iou).to(device)

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)        # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss



class Darknet(nn.Module):
    
    def __init__(self, num_classes, encoder_features, rm_dim):
        super(Darknet, self).__init__()

        self.num_classes = num_classes
        self.encoder = YoloEncoder(encoder_features)
        
        #self.shared_decoder = nn.Sequential()
        
        self.yolo_decoder = YoloDecoder(num_classes = num_classes)
        
        self.yolo_loss = YoloLoss(feature_size=S, num_bboxes=B, num_classes=num_classes, 
                                  lambda_coord=l_coord, lambda_noobj = l_noobj)
        
        #self.rm_decoder = RmDecoder(rm_dim)
        
    def encode(self, x):
        
        # get all the representations laid out like this
        x = torch.cat([self.encoder(x[:, i, :]) for i in range(6)], dim = 1)
            
            
        #convert from dense representation from encoder into an image
        # x.view(...)
        
        #x = self.shared_decoder(x)
        
        return x
    
    def forward(self, x, yolo_targets = None):
        encoding = self.encode(x)
        
        bbox, yolo_loss = self.get_bounding_boxes(x, encoding = encoding, targets = yolo_targets)

        return bbox, yolo_loss

    
    # for easy use for competition
    # in competition, encoding is None
    def get_bounding_boxes(self, x, encoding = None, targets = None):
        if encoding is None:
            encoding = self.encode(x)
        
        outputs = self.yolo_decoder(encoding)
        
        if targets is not None:
            yoloLossValue = self.yolo_loss(outputs, targets)
        else:
            yoloLossValue = 0
        
        
        boxes = []
        
        for output in outputs:
            # Get detected boxes_detected, labels, confidences, class-scores.
            boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = decode_pred(output)
            if boxes_normalized_all.size(0) == 0:
                continue

            # Apply non maximum supression for boxes of each class.
            boxes_normalized, class_labels, probs = [], [], []

            for class_label in range(self.num_classes):
                mask = (class_labels_all == class_label)
                if torch.sum(mask) == 0:
                    continue # if no box found, skip that class.

                boxes_normalized_masked = boxes_normalized_all[mask]
                class_labels_maked = class_labels_all[mask]
                confidences_masked = confidences_all[mask]
                class_scores_masked = class_scores_all[mask]

                ids = nms(boxes_normalized_masked, confidences_masked)

                boxes_normalized.append(boxes_normalized_masked[ids])
                class_labels.append(class_labels_maked[ids])
                probs.append(confidences_masked[ids] * class_scores_masked[ids])

            boxes_normalized = torch.cat(boxes_normalized, 0)
            class_labels = torch.cat(class_labels, 0)
            probs = torch.cat(probs, 0)
        

            better_coordinates = FloatTensor(boxes_normalized.shape[0], 2, 4)
            translation = FloatTensor(boxes_normalized.shape[0], 2, 4)
            translation[:, 0, :].fill_(-40)
            translation[:, 1, :].fill_(40)

            center_x = (boxes_normalized[:, 0] + boxes_normalized[:, 2]) / 2 * WIDTH
            center_y = (boxes_normalized[:, 1] + boxes_normalized[:, 3]) / 2 * HEIGHT
            width = (boxes_normalized[:, 2] - boxes_normalized[:,0]) * WIDTH
            height = (boxes_normalized[:, 3] - boxes_normalized[:,1]) * HEIGHT
            
            x1 = center_x - width/2
            x2 = center_x + width/2
            x3 = center_x - width/2
            x4 = center_x + width/2
            
            y1 = center_y - height/2
            y2 = center_y + height/2
            y3 = center_y + height/2
            y4 = center_y - height/2
            
            
            better_coordinates[:, 0, 0] = x1
            better_coordinates[:, 0, 1] = x3
            better_coordinates[:, 0, 2] = x2
            better_coordinates[:, 0, 3] = x4
            
            better_coordinates[:, 1, 0] = y1
            better_coordinates[:, 1, 1] = y2
            better_coordinates[:, 1, 2] = y4
            better_coordinates[:, 1, 3] = y3
            
            better_coordinates[:, 1, :].mul_(-1)
            better_coordinates += translation
            
            boxes.append(better_coordinates)

        return tuple(boxes), yoloLossValue

    def init_weights(self, pretrained = ''):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    

