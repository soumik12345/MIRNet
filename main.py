from glob import glob
from mirnet.train import LowLightTrainer
from mirnet.utils import init_wandb, download_dataset


download_dataset('LOL')

init_wandb(
    project_name='mirnet', experiment_name='LOL_lowlight',
    wandb_api_key='cf0947ccde62903d4df0742a58b8a54ca4c11673'
)

train_low_light_images = glob('./our485/low/*')
train_high_light_images = glob('./our485/high/*')
valid_low_light_images = glob('./eval15/low/*')
valid_high_light_images = glob('./eval15/high/*')

trainer = LowLightTrainer()
trainer.build_dataset(
    train_low_light_images, train_high_light_images,
    valid_low_light_images, valid_high_light_images,
    crop_size=128, batch_size=16
)

trainer.compile()

trainer.train(epochs=100, checkpoint_dir='./checkpoints')
