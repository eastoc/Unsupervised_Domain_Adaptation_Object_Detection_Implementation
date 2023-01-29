_base_ = [
    '../_base_/models/faster_rcnn_r50_torch_maf.py', '../_base_/datasets/d2n.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
#optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
#lr_config = dict(policy='step', step=[10])

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[10])

# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=14)  # actual epoch = 4 * 3 = 12
checkpoint_config = dict(interval=1)
#cudnn_benchmark = False
#evaluation = dict(interval=100, metric='mAP', pre_eval=True)