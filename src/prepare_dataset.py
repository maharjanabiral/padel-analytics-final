import json
import numpy as np
import torch
from torch.utils.data import Dataset

seq_len = 20
classes = ["backhand", "forehand", "ready_position", "serve"]


def add_noise(x, std=0.01):
    noise = np.random.normal(0, std, x.shape)
    return x + noise


def random_scale(x, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return x * scale


def temporal_shift(x, max_shift=2):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=0)


def joint_dropout(x, drop_prob=0.1):
    mask = np.random.rand(*x.shape) > drop_prob
    return x * mask


def time_jitter(x):
    idx = np.random.permutation(x.shape[0])
    return x[idx]


def augment(x):

    if np.random.rand() < 0.5:
        x = add_noise(x, std=0.01)

    if np.random.rand() < 0.3:
        x = random_scale(x)

    if np.random.rand() < 0.3:
        x = temporal_shift(x, max_shift=2)

    if np.random.rand() < 0.2:
        x = joint_dropout(x, drop_prob=0.1)

    if np.random.rand() < 0.2:
        x = time_jitter(x)

    return x


class PadelDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.x = []
        self.y = []
        self.split = split

        for idx, class_name in enumerate(classes):
            annotation_path = f"data/annotations/{class_name}.json"

            with open(annotation_path, "r") as f:
                data = json.load(f)

            annotations = data["annotations"]
            annotations = sorted(annotations, key=lambda x: x["image_id"])

            frames = []

            for ann in annotations:
                keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(18, 3)

                missing = keypoints[:, 2] == 0
                keypoints[missing, 0] = 0
                keypoints[missing, 1] = 0

                left_hip = keypoints[11][:2]
                right_hip = keypoints[12][:2]
                hip_center = (left_hip + right_hip) / 2
                keypoints[:, :2] -= hip_center

                keypoints[:, 0] /= 1280.0
                keypoints[:, 1] /= 720.0

                frames.append(keypoints.flatten())

            frames = np.array(frames, dtype=np.float32)

            split_idx = int(len(frames) * 0.8)

            if split == "train":
                frames = frames[:split_idx]
            else:
                frames = frames[split_idx:]

            stride = 2

            for i in range(0, len(frames) - seq_len, stride):
                seq = frames[i:i + seq_len]

                self.x.append(seq)
                self.y.append(idx)

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index].copy()
        y = self.y[index]

        if self.split == "train":
            x = augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)