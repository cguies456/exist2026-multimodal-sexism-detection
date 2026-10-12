from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .data import MemeDataset, build_sensor_schema, label_vocab
from .models import CrossAttentionFusionModel, GatedFusionModel

TEXT_MODELS = {"en": "xlm-roberta-base", "es": "pysentimiento/robertuito-base-uncased"}
IMAGE_MODEL = "google/vit-base-patch16-224"

FUSION_CONFIGS = {
    "cross_attention": {"batch_size": 8, "lr": 5e-5, "epochs": 5, "clip_grad": 1.0, "sensor_mode": "padded"},
    "gated": {"batch_size": 2, "lr": 2e-5, "epochs": 3, "clip_grad": None, "sensor_mode": "schema"},
}
FINAL_FUSION = {"task1": "gated", "task2": "cross_attention", "task3": "gated"}
BINARY_NAMES = {"task1": ("NO", "YES"), "task2": ("JUDGEMENTAL", "DIRECT")}


@dataclass
class TrainedModel:
    model: torch.nn.Module
    tokenizer: object
    image_processor: object
    task: str
    fusion: str
    mode: str
    classes: list = None
    schema: list = None
    history: list = field(default_factory=list)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _output_head(task, fusion, num_classes):
    if task == "task3":
        return num_classes, "multilabel"
    return (2, "softmax") if fusion == "cross_attention" else (1, "sigmoid")


def _loss_and_predictions(logits, labels, mode):
    if mode == "softmax":
        return F.cross_entropy(logits, labels.long()), logits.argmax(dim=1)
    if mode == "sigmoid":
        logits = logits.squeeze(1)
        return F.binary_cross_entropy_with_logits(logits, labels.float()), (torch.sigmoid(logits) > 0.5).long()
    return F.binary_cross_entropy_with_logits(logits, labels), (torch.sigmoid(logits) > 0.5).long()


def _to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, mode, device, optimizer=None, clip_grad=None):
    training = optimizer is not None
    model.train(training)
    total_loss, targets, predictions = 0.0, [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch.pop("label").to(device)
            loss, preds = _loss_and_predictions(model(**_to_device(batch, device)), labels, mode)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            total_loss += loss.item()
            targets.append(labels.cpu())
            predictions.append(preds.cpu())

    targets = torch.cat(targets).long().numpy()
    predictions = torch.cat(predictions).numpy()
    average = "macro" if mode == "multilabel" else "binary"
    return total_loss / len(loader), f1_score(targets, predictions, average=average, zero_division=0)


def train_task(train_df, dev_df, lang, task, fusion, image_dir, epochs=None):
    config = FUSION_CONFIGS[fusion]
    device = _device()
    text_model = TEXT_MODELS[lang]
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)

    classes = label_vocab(train_df) if task == "task3" else None
    schema = build_sensor_schema(train_df) if config["sensor_mode"] == "schema" else None
    num_outputs, mode = _output_head(task, fusion, len(classes or []))

    def loader(df, shuffle):
        dataset = MemeDataset(df, tokenizer, image_processor, image_dir, config["sensor_mode"], task, classes, schema)
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)

    train_loader, dev_loader = loader(train_df, True), loader(dev_df, False)

    if fusion == "cross_attention":
        model = CrossAttentionFusionModel(text_model, IMAGE_MODEL, num_outputs)
    else:
        model = GatedFusionModel(text_model, IMAGE_MODEL, len(schema), num_outputs)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"])

    trained = TrainedModel(model, tokenizer, image_processor, task, fusion, mode, classes, schema)
    for epoch in range(epochs or config["epochs"]):
        train_loss, train_f1 = run_epoch(model, train_loader, mode, device, optimizer, config["clip_grad"])
        dev_loss, dev_f1 = run_epoch(model, dev_loader, mode, device)
        trained.history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_f1": train_f1,
                                "dev_loss": dev_loss, "dev_f1": dev_f1})
        print(f"[{lang} | {task} | {fusion}] epoch {epoch + 1}: "
              f"train loss {train_loss:.4f}, F1 {train_f1:.4f} | dev loss {dev_loss:.4f}, F1 {dev_f1:.4f}")
    return trained


def predict(trained, df, image_dir, batch_size=8):
    sensor_mode = FUSION_CONFIGS[trained.fusion]["sensor_mode"]
    dataset = MemeDataset(df, trained.tokenizer, trained.image_processor, image_dir, sensor_mode, schema=trained.schema)
    device = next(trained.model.parameters()).device
    trained.model.eval()
    with torch.no_grad():
        logits = torch.cat([trained.model(**_to_device(batch, device)).cpu()
                            for batch in DataLoader(dataset, batch_size=batch_size)])

    if trained.mode == "multilabel":
        probabilities = torch.sigmoid(logits)
        labels = []
        for row in probabilities:
            selected = (row > 0.5).nonzero().flatten().tolist() or [int(row.argmax())]
            labels.append([trained.classes[i] for i in selected])
        return labels

    positive = logits.argmax(dim=1) if trained.mode == "softmax" else (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
    names = BINARY_NAMES[trained.task]
    return [names[i] for i in positive.tolist()]
