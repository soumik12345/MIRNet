from glob import glob
import tensorflow as tf
from matplotlib import pyplot as plt
from mirnet.model import mirnet_model
from mirnet.dataloaders import LOLDataLoader


def test_output_dim():
    mirnet = mirnet_model(256, 3, 2, 64)
    x = tf.ones((1, 256, 256, 3))
    y = mirnet(x)
    assert x.shape == y.shape


def test_dataloader():
    lowlight_images = glob('./data/LOLdataset/our485/low/*')
    highlight_images = glob('./data/LOLdataset/our485/high/*')
    dataset = LOLDataLoader(
        images_lowlight=lowlight_images,
        images_highlight=highlight_images
    ).build_dataset(
        image_crop_size=128, batch_size=1, apply_transforms=True
    )
    print(dataset)
    x, y = next(iter(dataset))
    print(x.shape, y.shape)
    plt.imshow(tf.cast(x[0] * 255, dtype=tf.uint8))
    plt.title('Low Light Patch (128 x 128)')
    plt.show()
    plt.title('High Light Patch (128 x 128)')
    plt.imshow(tf.cast(y[0] * 255, dtype=tf.uint8))
    plt.show()


if __name__ == '__main__':
    # test_dataloader()
    test_output_dim()
