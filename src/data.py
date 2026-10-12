import ast
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DEV_PART = 0.2
SEED = 42
EEG_DIM = 144
BIO_DIM = 96
BASE_COLUMNS = ["id_EXIST", "text", "meme", "path_memes", "sensorial"]
TASK_LABELS = {"task1": "label_1", "task2": "label_2", "task3": "label_3"}


def load_memes(json_path):
    return pd.read_json(json_path).T


def majority_yes_no(votes):
    yes, no = votes.count("YES"), votes.count("NO")
    return "YES" if yes > no else "NO" if no > yes else "NA"


def majority_direct_judgemental(votes):
    direct, judgemental = votes.count("DIRECT"), votes.count("JUDGEMENTAL")
    return "DIRECT" if direct > judgemental else "JUDGEMENTAL" if judgemental > direct else "NA"


def union_of_categories(votes):
    labels = set()
    for annotator_labels in votes:
        if annotator_labels not in (["-"], []):
            labels |= set(annotator_labels)
    return sorted(labels) if labels else "NA"


def build_splits(df):
    """Returns {(lang, task): (train_df, dev_df)} with one hard label per meme, dropping memes without consensus."""
    df = df.copy()
    df["label_1"] = df["labels_task2_1"].apply(majority_yes_no)
    df["label_2"] = df["labels_task2_2"].apply(majority_direct_judgemental)
    df["label_3"] = df["labels_task2_3"].apply(union_of_categories)

    splits = {}
    for lang in ("en", "es"):
        data = df[df["lang"] == lang][BASE_COLUMNS + list(TASK_LABELS.values())]
        train, dev = train_test_split(data, test_size=DEV_PART, random_state=SEED)
        for task, col in TASK_LABELS.items():
            has_label = lambda d: d[d[col].apply(lambda v: v != "NA")][BASE_COLUMNS + [col]]
            splits[(lang, task)] = (has_label(train), has_label(dev))
    return splits


def label_vocab(df):
    return sorted({label.strip() for labels in df["label_3"] for label in _as_list(labels)})


def _as_list(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return value if isinstance(value, list) else []


def parse_sensorial(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(value)
            except (ValueError, SyntaxError):
                pass
    return {}


def _mean_over_users(modality):
    by_user = (modality or {}).get("by_user", {})
    if not by_user:
        return []
    keys = sorted({key for features in by_user.values() for key in features})
    vectors = [[features.get(k) if features.get(k) is not None else 0.0 for k in keys] for features in by_user.values()]
    return np.mean(vectors, axis=0).tolist()


def _fixed_length(values, dim):
    tensor = torch.tensor(values, dtype=torch.float32) if values else torch.zeros(1)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return F.pad(tensor, (0, dim - len(tensor))) if len(tensor) < dim else tensor[:dim]


def padded_sensor_features(sensorial):
    """Cross-attention input: EEG vector (144) and eye-tracking + heart-rate vector (96), averaged over viewers."""
    modalities = parse_sensorial(sensorial).get("modalities", {})
    eeg = _mean_over_users(modalities.get("EEG", {}))
    bio = _mean_over_users(modalities.get("ET", {})) + _mean_over_users(modalities.get("HR", {}))
    return _fixed_length(eeg, EEG_DIM), _fixed_length(bio, BIO_DIM)


def build_sensor_schema(df):
    keys = set()
    for value in df["sensorial"]:
        for modality, data in parse_sensorial(value).get("modalities", {}).items():
            for features in data.get("by_user", {}).values():
                if isinstance(features, dict):
                    keys.update(f"{modality}_{k}" for k in features)
    return sorted(keys) or ["dummy_feature"]


def schema_sensor_features(sensorial, schema):
    """Gated-fusion input: one value per modality_feature in the training schema, averaged over viewers."""
    means = {}
    for modality, data in parse_sensorial(sensorial).get("modalities", {}).items():
        values = {}
        for features in data.get("by_user", {}).values():
            if isinstance(features, dict):
                for key, value in features.items():
                    if value is not None:
                        values.setdefault(key, []).append(value)
        for key, vals in values.items():
            means[f"{modality}_{key}"] = sum(vals) / len(vals)
    return torch.tensor([means.get(key, 0.0) for key in schema], dtype=torch.float32)


def encode_label(value, task, classes=None):
    if task == "task1":
        return torch.tensor(int(str(value).strip().upper() == "YES"))
    if task == "task2":
        return torch.tensor(int(str(value).strip().upper() == "DIRECT"))
    target = torch.zeros(len(classes), dtype=torch.float32)
    index = {label: i for i, label in enumerate(classes)}
    for label in _as_list(value):
        if label.strip() in index:
            target[index[label.strip()]] = 1.0
    return target


class MemeDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, image_dir, sensor_mode,
                 task=None, classes=None, schema=None, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.sensor_mode = sensor_mode
        self.task = task
        self.classes = classes
        self.schema = schema
        self.max_length = max_length

        missing = sum(not os.path.exists(os.path.join(image_dir, p)) for p in self.df["path_memes"])
        if missing:
            print(f"Warning: {missing}/{len(self.df)} images not found in '{image_dir}'; a black image is used instead.")

    def __len__(self):
        return len(self.df)

    def _image(self, relative_path):
        try:
            return Image.open(os.path.join(self.image_dir, relative_path)).convert("RGB")
        except OSError:
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.tokenizer(str(row["text"]), padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        image = self.image_processor(images=self._image(row["path_memes"]), return_tensors="pt")
        item = {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "pixel_values": image["pixel_values"].squeeze(0),
        }
        if self.sensor_mode == "padded":
            item["eeg_features"], item["bio_features"] = padded_sensor_features(row["sensorial"])
        else:
            item["sensor_features"] = schema_sensor_features(row["sensorial"], self.schema)
        if self.task is not None:
            item["label"] = encode_label(row[TASK_LABELS[self.task]], self.task, self.classes)
        return item
