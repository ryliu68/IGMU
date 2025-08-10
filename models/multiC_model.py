from torchvision import transforms
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import warnings
from PIL import Image
from torch.nn import MultiheadAttention

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDatasetForInference(Dataset):
    def __init__(self, image_dir_or_list=None, data_type="CLIP"):

        if data_type == "CLIP":
            self.mean = (0.4815, 0.4578, 0.4082)
            self.std = (0.2686, 0.2613, 0.2758)
        else:
            raise ValueError(data_type)

        if isinstance(image_dir_or_list, str):
            self.image_paths = [os.path.join(image_dir_or_list, fname)
                                for fname in os.listdir(image_dir_or_list)
                                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        else:
            self.image_paths = image_dir_or_list

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Error loading image: {path} - {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

        image_tensor = self.transform(image)
        return {"pixel_values": image_tensor, "image_path": path}


class Bi_MultiC(nn.Module):
    def __init__(self, hidden_dim=512, dropout_rate=0.3):
        super(Bi_MultiC, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        x = x * attention_weights
        logits = self.classifier(x)
        return logits


class Bi_MultiC_Classifier(nn.Module):
    def __init__(self, clip_model, classifier_head):
        super(Bi_MultiC_Classifier, self).__init__()
        self.clip_model = clip_model
        self.classifier_head = classifier_head
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values, labels=None):
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values)
        logits = self.classifier_head(clip_outputs)
        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits.squeeze(), labels)
        return {"logits": logits, "loss": loss}

    def predict(self, pixel_values):
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values)
        logits = self.classifier_head(clip_outputs)
        preds = torch.sigmoid(logits).squeeze().round().cpu().numpy()
        return preds


class Multi_MultiC(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.multihead = MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.feedforward = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        attn_weights = self.attn(x)
        x_attn = x * attn_weights
        x_mha, _ = self.multihead(x_attn.unsqueeze(1), x_attn.unsqueeze(1), x_attn.unsqueeze(1))
        x_mha = x_mha.squeeze(1)
        g = self.gate(x_attn)
        x_fused = g * x_attn + (1 - g) * x_mha
        x_ffn = self.feedforward(x_fused)
        x = x_fused + x_ffn
        h = self.mlp1(x)
        h = h + x
        h = self.mlp2(h)
        logits = self.output(h)
        return logits


class Multi_MultiC_Classifier(nn.Module):
    def __init__(self, clip_model, classifier_head):
        super(Multi_MultiC_Classifier, self).__init__()
        self.clip_model = clip_model
        self.classifier_head = classifier_head
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values)
        logits = self.classifier_head(clip_outputs)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.long())
        return {"logits": logits, "loss": loss}

    def predict(self, pixel_values):
        with torch.no_grad():
            clip_outputs = self.clip_model.get_image_features(pixel_values)
        logits = self.classifier_head(clip_outputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        return preds
