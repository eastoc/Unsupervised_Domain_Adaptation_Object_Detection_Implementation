# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch.nn as nn
import torch
from ..losses import FocalLoss
from ..roi_heads.instance_da import InstanceAlignmentHead
import torch.nn.functional as F
from ..utils import cluster

@DETECTORS.register_module()
class DAFasterRCNN_Tri(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(DAFasterRCNN_Tri, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_fl = FocalLoss()
        
        #self.local_da = InstanceAlignmentHead()
        #self.local_da._init_weights()

        self.local_da_fore = InstanceAlignmentHead()
        self.local_da_fore._init_weights()

        self.local_da_back = InstanceAlignmentHead()
        self.local_da_back._init_weights()

        self.global_align = True
        self.local_align = True
        self.patch_align = True

    def extract_feat_train(self, img, gt_domain):
        """Directly extract features from the backbone+neck."""
        x, global_loss, patch_loss_bottom = self.backbone.forward_train(img, gt_domain)
        if self.with_neck:
            x = self.neck(x)
        return x, global_loss, patch_loss_bottom

    def adaptive_loss_weights(self, global_loss=0, patch_mid_loss=0, patch_bot_loss=0, local_loss=0):
        alpha = 0.1
        total = sum(global_loss)+patch_mid_loss+patch_bot_loss+local_loss
        global_lamdas = [alpha*loss/total for loss in global_loss]
        patch_mid_lamda = alpha*patch_mid_loss/total
        patch_bot_lamda = alpha*patch_bot_loss/total
        local_lamda = alpha*local_loss/total
        
        return global_lamdas, patch_mid_lamda, patch_bot_lamda, local_lamda

    def global_loss(self, domain_pred, gt_domain):
        da_globle_loss = torch.zeros(len(domain_pred))
        for i in range(len(domain_pred)):
            da_globle_loss[i] = self.criterion(domain_pred[i], gt_domain)
        lamda = adaptive_loss_weights(da_globle_loss)
        da_globle_losses = [lamda[i]*loss for i, loss in enumerate(da_globle_loss)]
        da_globle_losses = da_globle_loss.sum()

        return da_globle_losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_da=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        global_lamda = 0.1
        local_lamda = 0.1
    
        gt_domain = torch.zeros(len(gt_da), requires_grad=True, device='cuda').long()
        for i,index in enumerate(gt_da):
            if index == 1:
                gt_domain[i] = 1

        x, global_loss, patch_bottom_loss = self.extract_feat_train(img, gt_domain)

        losses = dict()

        # Domain forward

        # RPN forward and loss
        rpn_losses = None
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_da=gt_domain,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            if rpn_losses == None:
                rpn_losses=dict(
                    loss_rpn_cls=torch.tensor(0.0,requires_grad=False,device='cuda').float(),
                    loss_rpn_bbox=torch.tensor(0.0,requires_grad=False,device='cuda').float())
            losses.update(rpn_losses)
            #print(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses, bbox_feats, bbox_cls = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_da,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        
        losses.update(roi_losses)
       
        if self.local_align:
            local_da_loss = self.group_local_da_loss(bbox_feats, local_lamda, bbox_cls)

        global_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        patch_mid_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        patch_bot_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        local_lamda = torch.tensor(0.2, requires_grad=True, device='cuda')

        if self.global_align:
            da_globle_loss = global_lamda * sum(global_loss)
            losses.update(globle_da_loss=da_globle_loss)
        
        if self.patch_align:    
            patch_bot_loss = patch_bot_lamda * patch_bottom_loss
            losses.update(patch_bottom_loss=patch_bot_loss)
        
        if self.local_align:
            local_da_loss = local_lamda * local_da_loss
            losses.update(local_da_loss=local_da_loss)
        
        return losses

    def complete(self, feats, cls_score, k):
        cls_score = torch.cat(cls_score, dim=0)
        #print(cls_score)
        cls_score = F.softmax(cls_score, dim=-1)
        top_idx = torch.argmax(cls_score, dim=0)
        add_num = k - len(feats)
        add_feats = []
        for i in range(add_num):
            add_feats.append(feats[top_idx].unsqueeze(0))
        add_feats = torch.cat(add_feats, dim=0)
        feats = torch.cat([feats, add_feats], dim=0)
        #print(feats.size())
        return feats

    def group(self, feats, cls_score, k=10):
        """
        :param: feats: tensor[n x 1024]
        return: group_feats: tensor[k x 1024]
        """
        #print(cls_score)
        if len(feats) > k:
            kmeans = cluster(feats, k=k)
            kmeans.run()
            group_feats = kmeans.centrods.coord
            group_feats = torch.cat(group_feats, dim=0)
            return group_feats

        elif len(feats) == k:
            return feats

        elif len(feats) < k:
            feats = self.complete(feats, cls_score, k=k)
            return feats

    def group_local_da_loss(self, bbox_feats ,lamda, bbox_cls):
        # source
        fg_src = []# 源域前景特征
        bg_src = []# 源域背景特征
        fg_cls_score = []
        bg_cls_score = []
        for i, feat in enumerate(bbox_feats[0]):
            cls_temp = F.softmax(bbox_cls[0][i], dim=-1)
            if cls_temp[0] >=0.5:
            #if torch.argmax(bbox_cls[0][i]) == 0:
                fg_src.append(feat.unsqueeze(0))
                fg_cls_score.append(cls_temp[0].unsqueeze(0))
            else:
                bg_src.append(feat.unsqueeze(0))
                bg_cls_score.append(cls_temp[1].unsqueeze(0))
        if len(fg_src)!=0:
            fg_src = torch.cat(fg_src, dim=0)
            fg_src = self.group(fg_src, fg_cls_score)
        if len(bg_src)!=0:
            bg_src = torch.cat(bg_src, dim=0)
            bg_src = self.group(bg_src, bg_cls_score)

        fg_src_gt = torch.zeros(len(fg_src), requires_grad=False, device='cuda').long()
        bg_src_gt = torch.zeros(len(bg_src), requires_grad=False, device='cuda').long()
        # target
        fg_tar = [] # 目标域前景特征
        bg_tar = [] # 目标域背景特征
        fg_cls_score = []
        bg_cls_score = []
        for i, feat in enumerate(bbox_feats[1]):
            cls_temp = F.softmax(bbox_cls[1][i], dim=-1)
            if cls_temp[0] >= 0.5: 
            #if torch.argmax(bbox_cls[1][i]) == 0:
                fg_tar.append(feat.unsqueeze(0))
                fg_cls_score.append(cls_temp[0].unsqueeze(0))
            else:
                bg_tar.append(feat.unsqueeze(0))
                bg_cls_score.append(cls_temp[1].unsqueeze(0))
        
        if len(fg_tar)!=0:
            fg_tar = torch.cat(fg_tar, dim=0)
            fg_tar = self.group(fg_tar, fg_cls_score)
        if len(bg_tar)!=0:
            bg_tar = torch.cat(bg_tar, dim=0)
            bg_tar = self.group(bg_tar, bg_cls_score)
        fg_tar_gt = torch.ones(len(fg_tar), requires_grad=False, device='cuda').long()
        bg_tar_gt = torch.ones(len(bg_tar), requires_grad=False, device='cuda').long()

        # concatnate
        if len(fg_src)!=0 & len(fg_tar)!=0:
            fg_feat = torch.cat([fg_src, fg_tar], dim=0)
            fore_gt = torch.cat([fg_src_gt, fg_tar_gt], dim=0)
        elif len(fg_src)!=0:
            fg_feat = fg_src
            fore_gt = fg_src_gt
        elif len(fg_tar)!=0:
            fg_feat = fg_tar
            fore_gt = fg_tar_gt
        else:
            fg_feat = []

        if len(bg_src)!=0 & len(bg_tar)!=0:
            bg_feat = torch.cat([bg_src, bg_tar], dim=0)
            back_gt = torch.cat([bg_src_gt, bg_tar_gt], dim=0)
        elif len(bg_src)!=0:
            bg_feat = bg_src
            back_gt = bg_src_gt
        elif len(bg_tar)!=0:
            bg_feat = bg_tar
            back_gt = bg_tar_gt
        else:
            bg_feat = []
        # inference
        ins_loss_fore = torch.tensor(0.0, requires_grad=True, device='cuda').float()
        ins_loss_back = torch.tensor(0.0, requires_grad=True, device='cuda').float()
        if len(fg_feat)!=0:
            pred_da_fore = self.local_da_fore(fg_feat)
            ins_loss_fore = self.criterion_fl(pred_da_fore, fore_gt)
            
        if len(bg_feat)!=0:
            pred_da_back = self.local_da_back(bg_feat)
            ins_loss_back = self.criterion_fl(pred_da_back, back_gt)
           
        # loss
        ins_loss = ins_loss_fore.item() + ins_loss_back.item()
        
        return ins_loss

    def local_da_loss(self, bbox_feats, lamda):
        
        _local_da_loss = dict()
        label_src = torch.zeros(len(bbox_feats[0]), requires_grad=False, device='cuda').long()
        label_tar = torch.ones(len(bbox_feats[1]), requires_grad=False, device='cuda').long()
        label_da = torch.cat([label_src, label_tar], dim=0)
        bbox_feats = torch.cat(bbox_feats, dim=0)

        pred_da = self.local_da(bbox_feats)
        _ins_da_loss = self.criterion(pred_da, label_da)

        return _ins_da_loss

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x= self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs