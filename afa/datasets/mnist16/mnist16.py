import numpy as np
import tensorflow_datasets as tfds
from skimage.transform import resize
from tensorflow_datasets.image_classification import MNIST
from tensorflow_datasets.image_classification.mnist import (
    MNIST_NUM_CLASSES,
    _MNIST_CITATION,
    _extract_mnist_labels,
    _extract_mnist_images,
)


class Mnist16(MNIST):
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The MNIST database of handwritten digits."),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(16, 16, 1)),
                    "label": tfds.features.ClassLabel(num_classes=MNIST_NUM_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="http://yann.lecun.com/exdb/mnist/",
            citation=_MNIST_CITATION,
        )

    def _generate_examples(self, num_examples, data_path, label_path):
        """Generate MNIST examples as dicts.

        Args:
          num_examples (int): The number of example.
          data_path (str): Path to the data files
          label_path (str): Path to the labels

        Yields:
          Generator yielding the next examples
        """
        images = _extract_mnist_images(data_path, num_examples)
        labels = _extract_mnist_labels(label_path, num_examples)
        data = list(zip(images, labels))

        # Using index as key since data is always loaded in same order.
        for index, (image, label) in enumerate(data):
            record = {
                "image": (resize(image / 255.0, (16, 16)) * 255).astype(np.uint8),
                "label": label,
            }
            yield index, record
