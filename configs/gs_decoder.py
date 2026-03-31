"""Gaussian Decoder configuration for WorldSplat."""

# Training settings
max_steps = 100000
batch_size_train = 1
batch_size_test = 4
learning_rate = 1e-4
weight_decay = 0.01
warmup_iters = 500
save_interval = 5000
eval_interval = 1000

# Data settings
data_root = "data/nuscenes"
num_views = 6
image_size = (224, 400)
frame_num = 4
frame_interval = 2
near = 0.1
far = 1000.0
pc_range = [-50, -50, -3, 50, 50, 12]

# Model settings
model = dict(
    type="LatentGaussianDecoder",
    vae_pretrained="pretrained/sd-vae-ft-ema",
    vae_scale_factor=0.18215,
    near=near,
    far=far,
    pc_range=pc_range,
    # Loss weights
    weight_recon=1.0,
    weight_perceptual=0.05,
    weight_depth=0.01,
    weight_seg=0.1,
    perceptual_resolution=(224, 400),
    # Pixel decoder
    pixel_gs=dict(
        type="LatentDecoder_SD",
        gs_params_channels=17,  # 3(xyz) + 3(rgb) + 1(opacity) + 4(rotation) + 3(scale) + 2(seg) + 1(depth)
        num_views=num_views,
    ),
    # Gaussian renderer
    renderer=dict(
        type="GaussianRenderer",
        near=near,
        far=far,
        bg_color=(0.0, 0.0, 0.0),
    ),
)

# Dataset settings
train_dataset = dict(
    type="GSDecoderDataset",
    data_root=data_root,
    split="train",
    num_views=num_views,
    image_size=image_size,
    frame_num=frame_num,
    frame_interval=frame_interval,
)

val_dataset = dict(
    type="GSDecoderDataset",
    data_root=data_root,
    split="val",
    num_views=num_views,
    image_size=image_size,
    frame_num=frame_num,
    frame_interval=frame_interval,
)
