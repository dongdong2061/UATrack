class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/pretrained_networks'
        self.got10k_val_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/got10k/val'
        self.lasot_lmdb_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/coco_lmdb'
        self.coco_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/coco'
        self.lasot_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/lasot'
        self.got10k_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/got10k/train'
        self.trackingnet_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/trackingnet'
        self.depthtrack_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/depthtrack/train'
        self.lasher_dir = 'data/LasHeR'
        self.visevent_dir = '/data/wuyingjie/dzd/TUMFNet_extended/TUMFNet/data/visevent/train'
