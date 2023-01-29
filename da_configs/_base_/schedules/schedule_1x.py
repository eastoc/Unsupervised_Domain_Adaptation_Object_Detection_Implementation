# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[6, 8])
runner = dict(type='EpochBasedRunner', max_epochs=10)
#runner = dict(type='IterBasedRunner', max_iters=50000)
