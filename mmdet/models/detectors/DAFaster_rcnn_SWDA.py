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
class FasterRCNN_SWDA(TwoStageDetector):
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
        super(FasterRCNN_SWDA, self).__init__(
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

        self.global_align = True
        self.context = True
        self.patch_align = True

    def extract_feat_train(self, img, gt_domain):
        """Directly extract features from the backbone+neck."""
        if self.context == False:
            x, global_loss, patch_loss_bottom = self.backbone.forward_train(img, gt_domain)
            if self.with_neck:
                x = self.neck(x)
            return x, global_loss, patch_loss_bottom
        elif self.context == True:
            x, global_loss, patch_loss_bottom, context_feats = self.backbone.forward_train(img, gt_domain)
            if self.with_neck:
                x = self.neck(x)
            return x, global_loss, patch_loss_bottom, context_feats

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
        if self.context==False:
            x, global_loss, patch_bottom_loss = self.extract_feat_train(img, gt_domain)
        else:
            x, global_loss, patch_bottom_loss, context_feats = self.extract_feat_train(img, gt_domain)

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
        if self.context ==False:
            roi_losses, bbox_feats, bbox_cls = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_da,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        else:
            roi_losses, bbox_feats, bbox_cls = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels, gt_da,
                                                    gt_bboxes_ignore, gt_masks,context_feats,
                                                    **kwargs)
        
        #roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
        #                                         gt_bboxes, gt_labels,
        #                                         gt_bboxes_ignore, gt_masks,
        #                                         **kwargs)
        
        losses.update(roi_losses)
        """
        if self.global_align:
            da_img_losses.update(da_globle_loss=da_globle_losses)
            #losses.update(da_img_losses)

        """

        #global_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        #patch_bot_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        
        da_globle_loss = 0.1 * global_loss
        losses.update(globle_da_loss=da_globle_loss)
        
        patch_bot_loss = 0.1 * patch_bottom_loss     
        losses.update(patch_bottom_loss=patch_bot_loss)
        
        return losses

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
        if self.context == False:
            x= self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x
        elif self.context == True:
            x, context_feats = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x, context_feats

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


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.context==True:
            x, context_feats = self.extract_feat(img)
        else:
            x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
