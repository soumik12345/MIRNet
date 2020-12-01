import os
import gdown
import wandb
import subprocess
import tensorflow as tf
from matplotlib import pyplot as plt


def psnr(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


def init_wandb(project_name, experiment_name, wandb_api_key):
    """Initialize Wandb
    Args:
        project_name: project name on Wandb
        experiment_name: experiment name on Wandb
        wandb_api_key: Wandb API Key
    """
    if project_name is not None and experiment_name is not None:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(project=project_name, name=experiment_name)


def download_dataset(dataset_tag):
    """Utility for downloading and unpacking dataset dataset
    Args:
        dataset_tag: Tag for the respective dataset.
        Available tags -> ('LOL')
    """
    print('Downloading dataset...')
    if dataset_tag == 'LOL':
        gdown.download(
            'https://drive.google.com/uc?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB',
            'LOLdataset.zip', quiet=False
        )
        print('Unpacking Dataset')
        subprocess.run(['unzip', 'LOLdataset.zip'])
        print('Done!!!')
    else:
        raise AssertionError('Dataset tag not found')


def plot_result(image, enhanced):
    """Utility for Plotting inference result
    Args:
        image: original image
        enhanced: enhanced image
    """
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 2, 1).set_title('Original Image')
    _ = plt.imshow(image)
    fig.add_subplot(1, 2, 2).set_title('Enhanced Image')
    _ = plt.imshow(enhanced)
    plt.show()
