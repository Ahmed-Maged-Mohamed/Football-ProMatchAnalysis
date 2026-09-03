"""Team clustering using SigLIP embeddings, UMAP and K-Means."""

from typing import Iterable

import cv2
import numpy as np


def central_jersey_crop(frame: np.ndarray, box: Iterable[float]) -> np.ndarray:
    """Crop the central torso while excluding grass, shorts and nearby players."""
    x1, y1, x2, y2 = map(int, box)
    height, width = frame.shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    left = max(0, x1 + int(0.20 * bw))
    right = min(width, x2 - int(0.20 * bw))
    top = max(0, y1 + int(0.10 * bh))
    bottom = min(height, y1 + int(0.55 * bh))
    return frame[top:bottom, left:right]


class SiglipTeamClassifier:
    """Two-team appearance classifier fitted without team labels."""

    model_name = "google/siglip-base-patch16-224"

    def __init__(self, device: str = "cpu", batch_size: int = 32, random_state: int = 42):
        try:
            import torch
            import umap
            from sklearn.cluster import KMeans
            from transformers import AutoImageProcessor, SiglipVisionModel
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP team clustering requires transformers, umap-learn and scikit-learn. "
                "Install requirements-advanced.txt."
            ) from exc
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        # Image-only processor: loading AutoProcessor also initializes the
        # unused text tokenizer and unnecessarily requires protobuf.
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.encoder = SiglipVisionModel.from_pretrained(self.model_name).to(device).eval()
        self.reducer = umap.UMAP(n_components=3, n_neighbors=15, random_state=random_state)
        self.clusterer = KMeans(n_clusters=2, n_init=20, random_state=random_state)
        self.fitted = False
        self.training_projection = None
        self.training_labels = None

    def _features(self, crops: list[np.ndarray]) -> np.ndarray:
        valid = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops if c.size]
        if not valid:
            return np.empty((0, 768), dtype=np.float32)
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(valid), self.batch_size):
                inputs = self.processor(images=valid[start:start + self.batch_size], return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.encoder(**inputs).last_hidden_state.mean(dim=1)
                output = self.torch.nn.functional.normalize(output, dim=1)
                batches.append(output.cpu().numpy())
        return np.concatenate(batches)

    def fit(self, crops: list[np.ndarray]) -> None:
        features = self._features(crops)
        if len(features) < 10:
            raise ValueError("At least 10 valid player crops are required to fit team clusters.")
        projections = self.reducer.fit_transform(features)
        self.clusterer.fit(projections)
        self.training_projection = projections
        self.training_labels = self.clusterer.labels_.copy()
        self.fitted = True

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Team classifier has not been fitted.")
        features = self._features(crops)
        if len(features) == 0:
            return np.empty(0, dtype=int)
        return self.clusterer.predict(self.reducer.transform(features))

    def save_embedding_plot(self, path: str) -> None:
        """Save the fitted three-dimensional UMAP clusters as a 2D diagnostic."""
        if not self.fitted or self.training_projection is None:
            raise RuntimeError("Team classifier has not been fitted.")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 6))
        scatter = axis.scatter(
            self.training_projection[:, 0],
            self.training_projection[:, 1],
            c=self.training_labels,
            cmap="coolwarm",
            s=28,
            alpha=.85,
        )
        axis.set(title="SigLIP jersey embeddings — UMAP + K-Means", xlabel="UMAP 1", ylabel="UMAP 2")
        axis.grid(alpha=.2)
        figure.colorbar(scatter, ax=axis, ticks=[0, 1], label="team cluster")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
