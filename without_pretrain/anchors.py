import torch 
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
from boxes import box_iou, nms

# calculate the offset between actual coordinates and anchor boxes
def get_offsets(anchor_boxes, actual_boxes):
    actual_width = actual_boxes[:, 2] - actual_boxes[:, 0]
    actual_height = actual_boxes[:, 3] - actual_boxes[:, 1]
    actual_center_x = actual_boxes[:, 0] + 0.5*actual_width
    actual_center_y = actual_boxes[:, 1] + 0.5*actual_height

    gt_width = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    gt_height = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    gt_center_x = anchor_boxes[:, 0] + 0.5*gt_width
    gt_center_y = anchor_boxes[:, 1] + 0.5*gt_height

    delta_x = (gt_center_x - actual_center_x) / actual_width
    delta_y = (gt_center_y - actual_center_y) / actual_height
    delta_scaleX = torch.log(gt_width / actual_width)
    delta_scaleY = torch.log(gt_height / actual_height)

    offsets = torch.cat([delta_x.unsqueeze(0), 
                    delta_y.unsqueeze(0),
                    delta_scaleX.unsqueeze(0),
                    delta_scaleY.unsqueeze(0)],
                dim=0)
    return offsets.permute(1,0)


def get_bbox_gt(bboxes1, classes, anchor_boxes, sz):
    # ex1, ey1, ex2, ey2 are the four coordinates of fl, br
    bboxes = bboxes1.clone()
    bboxes *= 10
    bboxes = bboxes + 400
    classes += 1

    threshold1 = 0.7
    threshold2 = 0.3
    ex1 = bboxes[:, 0, 3].unsqueeze(0)
    ey1 = bboxes[:, 1, 3].unsqueeze(0)
    ex2 = bboxes[:, 0, 0].unsqueeze(0)
    ey2 = bboxes[:, 1, 0].unsqueeze(0)
    actual_boxes = torch.cat([ex1, ey1, ex2, ey2], dim=0)
    actual_boxes = actual_boxes.permute(1,0)

    actual_width = actual_boxes[:, 2] - actual_boxes[:, 0]
    actual_height = actual_boxes[:, 3] - actual_boxes[:, 1]
    actual_center_x = actual_boxes[:, 0] + 0.5*actual_width
    actual_center_y = actual_boxes[:, 1] + 0.5*actual_height
    
    gt_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    gt_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    gt_center_x = anchor_boxes[:, 0] + 0.5*gt_widths
    gt_center_y = anchor_boxes[:, 1] + 0.5*gt_heights

    ious = box_iou(anchor_boxes, actual_boxes)
    vals, inds = torch.max(ious, dim=1)
    gt_classes = torch.zeros((sz*sz*4)).type(torch.long)
    gt_offsets = torch.zeros((sz*sz*4, 4)).type(torch.double)

    gt_classes[vals > threshold1] = classes[inds[vals > threshold1]] # foreground anchors
    gt_classes[vals < threshold2] = 0 # background anchors
    gt_classes[(vals >= threshold2) & (vals < threshold1)] = -1 # anchors to ignore

    actual_box_es = actual_boxes[inds[vals > threshold1]]
    ref_boxes = anchor_boxes[vals > threshold1]
    g_offsets = get_offsets(ref_boxes, actual_box_es)
    gt_offsets[vals > threshold1] = g_offsets

    return gt_classes, gt_offsets
    

# transfer coordinates back to [N,2,4] format 
def Transform_coor(anchor_boxes, gt_offsets, gt_classes, nms_threshold=0.1, plot=False):
    inds = (gt_classes != 0)
    anchor_boxes = anchor_boxes[inds]
    gt_offsets = gt_offsets[inds]
    gt_classes = gt_classes[inds]

    delta_x = gt_offsets[:,0]
    delta_y = gt_offsets[:,1]
    delta_scaleX = gt_offsets[:,2]
    delta_scaleY = gt_offsets[:,3]
    gt_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    gt_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    gt_center_x = anchor_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = anchor_boxes[:, 1] + 0.5 * gt_heights

    ex_width = gt_widths / torch.exp(delta_scaleX)
    ex_height = gt_heights / torch.exp(delta_scaleY)
    ex_center_x = gt_center_x - delta_x*ex_width
    ex_center_y = gt_center_y - delta_y*ex_height

    ex1 = ex_center_x - 0.5*ex_width
    ex2 = ex_center_x + 0.5*ex_width
    ey1 = ex_center_y - 0.5*ex_height
    ey2 = ex_center_y + 0.5*ex_height

    pred_boxes = torch.cat([ex1.unsqueeze(0), ey1.unsqueeze(0), ex2.unsqueeze(0), ey2.unsqueeze(0)], dim=0).permute(1,0)
    pred_boxes = pred_boxes.type(torch.float32)
    gt_classes = gt_classes.type(torch.float32)

    inds = nms(pred_boxes, gt_classes, nms_threshold)
    pred_boxes = pred_boxes[inds]
    coordinate_list = []
    for box in pred_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        width = abs(x1 - x2)
        height = abs(y1 - y2)
        coordinate_list.append(torch.tensor([x2, x2, x1, x1, y2, y1, y2, y1]).view(-1, 4))
        
    coordinate_list = torch.stack(coordinate_list)

    if plot:
      fig,ax = plt.subplots(1)
      a = torch.zeros(800,800)
      ax.imshow(a)
      for box in pred_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rect = patches.Rectangle((x1,y1),abs(x1 - x2),abs(y1 - y2),linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

    return coordinate_list


