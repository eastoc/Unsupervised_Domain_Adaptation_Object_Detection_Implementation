# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch.nn as nn
import torch
from ..losses import FocalLoss
from ..roi_heads.instance_da import InstanceAlignmentHead
import torch.nn.functional as F
from ..utils import cluster
from mmcv.runner import auto_fp16

@DETECTORS.register_module()
class DAFasterRCNN_Org(TwoStageDetector):
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
        super(DAFasterRCNN_Org, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.criterion = nn.CrossEntropyLoss()
        
        self.local_da = InstanceAlignmentHead()
        self.local_da._init_weights()

        self.global_align = True
        self.local_align = True

    def extract_feat_train(self, img, gt_domain):
        """Directly extract features from the backbone+neck."""
        x, global_loss, img_feat = self.backbone.forward_train(img, gt_domain)
        if self.with_neck:
            x = self.neck(x)
        return x, global_loss, img_feat

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
    
        gt_domain = torch.zeros(len(gt_da), requires_grad=True, device='cuda').long()
        for i,index in enumerate(gt_da):
            if index == 1:
                gt_domain[i] = 1

        x, global_loss, imgs_feat = self.extract_feat_train(img, gt_domain)

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
        else:
            proposal_list = proposals

        roi_losses, bbox_feats, bbox_cls = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_da,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        
        #roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
        #                                         gt_bboxes, gt_labels,
        #                                         gt_bboxes_ignore, gt_masks,
        #                                         **kwargs)
        
        losses.update(roi_losses)
        #global_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        #local_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')
        local_lamda = 0.1
        consist_lamda = 0.1
        #consist_lamda = torch.tensor(0.1, requires_grad=True, device='cuda')

        if self.local_align:
            local_da_loss, ins_preds, ins_labels = self.local_da_loss(bbox_feats, local_lamda)
            local_da_loss = local_lamda * local_da_loss
            losses.update(local_da_loss=local_da_loss)

        if self.global_align:
            da_globle_loss = 0.1 * global_loss
            losses.update(globle_da_loss=da_globle_loss)

        consistency_loss = consist_lamda * self.consist_loss(imgs_feat, ins_preds, ins_labels)
        losses.update(consistency_loss=consistency_loss)
        
        return losses

    def consist_loss(self, imgs_feat, ins_preds, ins_labels):
        consistency_loss = 0
        for i, img_feat in enumerate(imgs_feat):
            img_logit = torch.sigmoid(imgs_feat)
            I = torch.nonzero(img_logit).size()[0] # the number of the activations in the feature map
            img_logit = img_logit.sum()/I
            for j, ins_label in enumerate(ins_labels):
                if ins_label == i:
                    ins_pred = ins_preds[j]
                    ins_logit = torch.sigmoid(ins_pred)
                    #print(img_logit, ins_logit)
                    dist = torch.dist(img_logit, ins_logit[i], p=2)
                    consistency_loss = consistency_loss + dist

        return consistency_loss
        
    def local_da_loss(self, bbox_feats, lamda):
        
        _local_da_loss = dict()
        label_src = torch.zeros(len(bbox_feats[0]), requires_grad=False, device='cuda').long()
        label_tar = torch.ones(len(bbox_feats[1]), requires_grad=False, device='cuda').long()
        label_da = torch.cat([label_src, label_tar], dim=0)
        bbox_feats = torch.cat(bbox_feats, dim=0)

        pred_da = self.local_da(bbox_feats)
        _ins_da_loss = self.criterion(pred_da, label_da)
        
        return _ins_da_loss, pred_da, label_da

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

    @auto_fp16(apply_to=('img', ))
    def forward_tSNE(self, img, img_metas, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        return self._forward_tSNE(img, img_metas, **kwargs)

    def _forward_tSNE(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test_tSNE(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.tSNE_test(imgs, img_metas, **kwargs)


    def tSNE_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats_tSNE(imgs)
        return x

    def extract_feats_tSNE(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat_tSNE(img) for img in imgs]

    def extract_feat_tSNE(self, img):
        """Directly extract features from the backbone+neck."""
        x= self.backbone(img)
        
        return x

    def simple_test_tSNE(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test_tSNE(
            x, proposal_list, img_metas, rescale=rescale)