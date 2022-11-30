import os
import pickle

import gdown
import tensorflow as tf
import tensorflow_datasets as tfds

url="https://drive.google.com/file/d/1A5m9ZWaOq17AVIc8edSl9u6GmZBoWMZm/view?usp=sharing"

class Molecule_small(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Chemistry Data with 4 Features (3 important features + 1 useless feature)",
            features=tfds.features.FeaturesDict(
                {
                    "features": tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                    "label": tfds.features.ClassLabel(num_classes=2),
                }
            ),
            supervised_keys=("features", "label"),
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        data_path = gdown.download(
            url=url, output=os.path.join(dl_manager.download_dir, "molecule_small.pkl"), fuzzy=True 
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
