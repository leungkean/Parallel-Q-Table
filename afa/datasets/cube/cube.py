from dataclasses import dataclass

import numpy as np
import tensorflow_datasets as tfds
from tensorflow import float32


def _generate_cube_data(num_features=20, data_points=20000, sigma=0.3, seed=123):
    assert num_features >= 10, "cube data must have >= 10 features"
    rng = np.random.RandomState(seed)
    clean_points = rng.binomial(1, 0.5, (data_points, 3))
    labels = np.dot(clean_points, np.array([1, 2, 4]))
    points = clean_points + rng.normal(0, sigma, (data_points, 3))
    features = rng.rand(data_points, num_features)

    for i in range(data_points):
        offset = labels[i]
        for j in range(3):
            features[i, offset + j] = points[i, j]

    features = np.array(features, np.float32)
    labels = np.array(labels, np.int32)

    indices = np.arange(data_points)
    rng.shuffle(indices)
    train_indices = indices[: int(data_points * 0.5)]
    validation_indices = indices[int(data_points * 0.5) : -int(data_points * 0.25)]
    test_indices = indices[-int(data_points * 0.25) :]

    return (
        (features[train_indices], labels[train_indices]),
        (features[validation_indices], labels[validation_indices]),
        (features[test_indices], labels[test_indices]),
    )


@dataclass
class CubeConfig(tfds.core.BuilderConfig):
    num_features: int = 20
    data_points: int = 20000
    sigma: float = 0.3
    seed: int = 123


class Cube(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    BUILDER_CONFIGS = [
        CubeConfig(
            name="default",
            description="Default Cube configuration with 20 features and 20000 total examples.",
            num_features=20,
            data_points=20000,
            sigma=0.3,
            seed=123,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "features": tfds.features.Tensor(
                        shape=(self.builder_config.num_features,), dtype=float32
                    ),
                    "label": tfds.features.ClassLabel(num_classes=8),
                }
            ),
            supervised_keys=("features", "label"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        train, val, test = _generate_cube_data(
            self.builder_config.num_features,
            self.builder_config.data_points,
            self.builder_config.sigma,
            self.builder_config.seed,
        )

        return {
            "train": self._generate_examples(train),
            "validation": self._generate_examples(val),
            "test": self._generate_examples(test),
        }

    def _generate_examples(self, data):
        for i, (x, y) in enumerate(zip(*data)):
            yield i, {
                "features": x,
                "label": y,
            }
