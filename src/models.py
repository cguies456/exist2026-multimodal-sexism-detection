import torch
import torch.nn as nn
from transformers import AutoModel

from .data import BIO_DIM, EEG_DIM


def _sensor_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 256), nn.ReLU(),
    )


def _classifier(hidden, num_outputs):
    return nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_outputs))


class CrossAttentionFusionModel(nn.Module):
    """Text tokens, image patches, EEG and biosignal tokens attend to each other; the text [CLS] token is classified.
    The pretrained text and image encoders are frozen."""

    def __init__(self, text_model, image_model, num_outputs):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.image_encoder = AutoModel.from_pretrained(image_model)
        hidden = self.text_encoder.config.hidden_size

        self.eeg_encoder = _sensor_mlp(EEG_DIM)
        self.bio_encoder = _sensor_mlp(BIO_DIM)
        self.eeg_proj = nn.Linear(256, hidden)
        self.bio_proj = nn.Linear(256, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, nhead=8, dropout=0.3, batch_first=True), num_layers=2
        )
        self.classifier = _classifier(hidden, num_outputs)

        for encoder in (self.text_encoder, self.image_encoder):
            for param in encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values, eeg_features, bio_features):
        text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        image = self.image_encoder(pixel_values=pixel_values).last_hidden_state
        eeg = self.eeg_proj(self.eeg_encoder(eeg_features)).unsqueeze(1)
        bio = self.bio_proj(self.bio_encoder(bio_features)).unsqueeze(1)

        tokens = self.layer_norm(torch.cat([text, image, eeg, bio], dim=1))
        attended, _ = self.cross_attention(tokens, tokens, tokens)
        fused = self.fusion_transformer(attended)
        return self.classifier(fused[:, 0])


class GatedFusion(nn.Module):
    """Learns a 0-1 weight per modality and returns the weighted sum of the three embeddings."""

    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, text, image, sensor):
        gates = self.gate(torch.cat([text, image, sensor], dim=-1))
        return gates[:, 0:1] * text + gates[:, 1:2] * image + gates[:, 2:3] * sensor


class GatedFusionModel(nn.Module):
    """Pooled text embedding, image [CLS] and projected sensor vector combined by a gate. Encoders are fine-tuned."""

    def __init__(self, text_model, image_model, sensor_dim, num_outputs):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.image_encoder = AutoModel.from_pretrained(image_model)
        hidden = self.text_encoder.config.hidden_size

        self.sensor_encoder = nn.Sequential(nn.Linear(sensor_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2))
        self.sensor_proj = nn.Linear(256, hidden)
        self.fusion = GatedFusion(hidden)
        self.classifier = _classifier(hidden, num_outputs)

    def forward(self, input_ids, attention_mask, pixel_values, sensor_features):
        text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image = self.image_encoder(pixel_values=pixel_values).last_hidden_state[:, 0]
        sensor = self.sensor_proj(self.sensor_encoder(sensor_features))
        return self.classifier(self.fusion(text, image, sensor))
