import os
import pickle

import gdown
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_ID = "1qZEVyNFD7IMPbGMDAI1tWOJ4d4RTrD0e"


class Gas10(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="The UCI Gas dataset with 10 features.",
            features=tfds.features.FeaturesDict(
                {
                    "features": tfds.features.Tensor(shape=(10,), dtype=tf.float32),
                    "label": tfds.features.ClassLabel(num_classes=3),
                }
            ),
            supervised_keys=("features", "label"),
            homepage="https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+#",
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        data_path = gdown.download(
            output=os.path.join(dl_manager.download_dir, "gas.pkl"), id=DATA_ID
        )

        with open(data_path, "rb") as fp:
            data = pickle.load(fp)

        return {
            "train": self._generate_examples(data["train"]),
            "validation": self._generate_examples(data["valid"]),
            "test": self._generate_examples(data["test"]),
        }

    def _generate_examples(self, data):
        for i, (x, y) in enumerate(zip(*data)):
            yield i, dict(features=x, label=y)
