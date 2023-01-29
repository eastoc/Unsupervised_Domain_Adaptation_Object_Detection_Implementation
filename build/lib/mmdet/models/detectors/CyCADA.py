# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch.nn as nn
import torch
from ..losses import FocalLoss

@DETECTORS.register_module()
class CyCADA(TwoStageDetector):
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
        super(CyCADA, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.gan_criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_fl = FocalLoss(gamma=3, alpha=0.5)

    def extract_feat_train(self, img, gt_da):
        """Directly extract features from the backbone+neck."""
        cycle_loss, dsn_loss = self.backbone.forward_train(img, gt_da)
        if self.with_neck:
            x = self.neck(x)
        return cycle_loss, dsn_loss

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
        
        cycle_loss, dsn_loss = self.extract_feat_train(img, gt_da)
        #cycle_loss, dsn_logits_s, dsn_logits_t = self.extract_feat_train(img, gt_da)
        
        gt_domain = torch.zeros(len(gt_da), requires_grad=True, device='cuda').long()
        for i,index in enumerate(gt_da):
            if index == 1:
                gt_domain[i] = 1
        
        #print('da',gt_da)
        #print('domain gt', gt_domain)
        losses = dict()

        # Domain forward
        
        # loss of the discriminator for GAN
        #print('dsn log',dsn_logits_s.size())
        #gt_gan = torch.tensor([[0.0,1.0],[1.0, 0.0]], requires_grad=True, device='cuda').float()
        #dsn_loss = torch.tensor(0.0, requires_grad=True, device='cuda').float()
        #dsn_loss = self.gan_criterion(dsn_logits_s, gt_gan) + self.gan_criterion(dsn_logits_t, gt_gan)
        
        losses.update(GAN_loss=dsn_loss)

        # loss of the global adverserial discriminator
        '''
        da_globle_loss = torch.zeros(3)
        for i in range(len(domain_pred)):
            print('domain pred:',domain_pred[i])
            print('domain gt', gt_domain)
            da_globle_loss[i] = criteria(domain_pred[i], gt_domain)
            #print('img',domain_pred[i].size(),gt_domain.size())
        da_globle_losses = global_lamda*da_globle_loss.sum()

        # patch domain alignment loss
        domain_loss_local = []
        for i, patch_feat in enumerate(patch_feats):
            if gt_domain[i] == 0:
                domain_loss_local.append(0.5 * torch.mean(torch.sigmoid(patch_feat) ** 2))
            elif gt_domain[i] == 1:
                domain_loss_local.append(0.5 * torch.mean(torch.sigmoid(1 - patch_feat) ** 2))

        da_globle_losses = da_globle_losses + sum(domain_loss_local)
        losses.update(da_globle_losses=da_globle_losses)
        '''
        #print('map: ', x[0].size())

        # RPN forward and loss
        """
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
                    loss_rpn_cls=torch.tensor(0.0,requires_grad=True,device='cuda').float(),
                    loss_rpn_bbox=torch.tensor(0.0,requires_grad=True,device='cuda').float())
            losses.update(rpn_losses)
            
        else:
            proposal_list = proposals

        
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_da,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)
        """
        cycle_loss = 0.1*cycle_loss
        losses.update(cycle_loss=cycle_loss)
        self.backbone.save_gan()
        return losses

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
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