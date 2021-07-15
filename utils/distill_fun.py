import torch
import numpy as np
def bbox_overlaps_batch(anchors, gt_boxes,imgsize):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,2:].contiguous() # have2，


        gt_boxes_x = gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1
        gt_boxes_y = gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view( batch_size,1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps
def generate_anchors(base_size, anchors):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    - [3,6, 5,11, 7,14]  # P3/8
    return shape=[3,4]
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    _, _, x_ctr, y_ctr = _whctrs(base_anchor)
    aim_ancher = []
    for anchor in anchors:

        x1 = x_ctr - 0.5 * base_size * anchor[0]
        y1 = y_ctr - 0.5 * base_size * anchor[1]
        x2 = x_ctr + 0.5 * base_size * anchor[0]
        y2 = y_ctr + 0.5 * base_size * anchor[1]
        aim_ancher.append([x1,y1,x2,y2])

    return np.array(aim_ancher)
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def make_gt_boxes(gt_boxes,max_num_box,batch,imgsize):
    new_gt_boxes = []
    for i in range(batch):
        boxes = gt_boxes[gt_boxes[:,0]==i]
        num_boxes = boxes.size(0)
        if num_boxes<max_num_box:
            gt_boxes_padding = torch.FloatTensor(max_num_box, gt_boxes.size(1)).zero_()
            gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
        else:
            #gt_boxes_padding = torch.FloatTensor(max_num_box, gt_boxes.size(1)).zero_()
            gt_boxes_padding = gt_boxes[:max_num_box]

        new_gt_boxes.append(gt_boxes_padding.unsqueeze(0))
    new_gt_boxes = torch.cat(new_gt_boxes)
    # x_c,y_c to x1y1x2y2
    new_gt_boxes_aim = new_gt_boxes
    new_gt_boxes_aim[:, :, 2] = (new_gt_boxes[:, :, 2] - 0.5 * new_gt_boxes[:,:,4] )* imgsize[1]
    new_gt_boxes_aim[:, :, 3] = (new_gt_boxes[:, :, 3] - 0.5 * new_gt_boxes[:, :, 5])* imgsize[0]
    new_gt_boxes_aim[:, :, 4] = (new_gt_boxes[:, :, 2] + 0.5 * new_gt_boxes[:, :, 4])* imgsize[1]
    new_gt_boxes_aim[:, :, 5] = (new_gt_boxes[:, :, 3] + 0.5 * new_gt_boxes[:, :, 5])* imgsize[0]
    return new_gt_boxes_aim
def getMask(batch_size,gt_boxes,imgsize,feat,anchors):

        # map of shape (..., H, W)

        gt_boxes = make_gt_boxes(gt_boxes, 100, 6,imgsize)


        feat_stride = imgsize[0]/feat.size(2)
        anchors = torch.from_numpy(generate_anchors(feat_stride,anchors))
        feat = feat.cpu()
        height, width = feat.size(2), feat.size(3)
        feat_height, feat_width = feat.size(2), feat.size(3)
        shift_x = np.arange(0, feat_width) * feat_stride
        shift_y = np.arange(0, feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(feat).float()

        A = anchors.size(0)
        K = shifts.size(0)

        anchors = anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)
        IOU_map = bbox_overlaps_batch(all_anchors, gt_boxes,imgsize).view(
                                 batch_size, height, width, A, gt_boxes.shape[1])

        mask_batch = []
        for i in range(batch_size):
                max_iou, _ = torch.max(IOU_map[i].view(height* width* A,
                                                       gt_boxes.shape[1]), dim = 0)
                mask_per_im = torch.zeros([height, width], dtype=torch.int64).cuda()
                for k in range(gt_boxes.shape[1]):
                    if torch.sum(gt_boxes[i][k]) == 0.:
                        break
                    max_iou_per_gt = max_iou[k]*0.5
                    mask_per_gt = torch.sum(IOU_map[i][:,:,:,k]>max_iou_per_gt,
                                                                       dim = 2)
                    mask_per_im +=mask_per_gt.cuda()
                mask_batch.append(mask_per_im)
        return mask_batch






def compute_mask_loss(mask_batch,student_feature,teacher_feature,imitation_loss_weigth):
    mask_list = []
    for mask in mask_batch:
        mask = (mask > 0).float().unsqueeze(0)
        mask_list.append(mask)
    mask_batch = torch.stack(mask_list, dim=0)
    norms = mask_batch.sum() * 2
    mask_batch_s = mask_batch.unsqueeze(4)
    #stu_feature_adap = feature_adap(student_feature)
    mask_batch_7 = torch.cat([mask_batch_s,mask_batch_s,mask_batch_s,
                              mask_batch_s,mask_batch_s,mask_batch_s,mask_batch_s],dim=-1)
    sup_loss = (torch.pow(teacher_feature - student_feature, 2) * mask_batch_7).sum() / norms
    sup_loss = sup_loss * imitation_loss_weigth

    return sup_loss
