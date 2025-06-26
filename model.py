import os
import warnings

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from transformers import Trainer

# Suppress all warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir, processor, batch_size=32):
    """
    Loads image data using torchvision's ImageFolder and applies the processor.

    Args:
        data_dir (str): Directory containing image data.
        processor (callable): Function or processor for image preprocessing.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: DataLoader for processed images.
    """
    dataset = datasets.ImageFolder(root=data_dir)
    processed_dataset = CustomImageFolderDataset(dataset, processor)
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, num_workers=32, shuffle=False)
    return dataloader


class CustomTrainer(Trainer):
    """
    Custom Trainer that saves only the classifier head parameters.
    """

    def save_model(self, output_dir=None, **kwargs):
        """
        Overrides the save_model method to save only the classifier head.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        classifier_head_path = os.path.join(output_dir, "classifier_head.pth")
        torch.save(self.model.classifier_head.state_dict(), classifier_head_path)

        # Optionally save processor if available
        if hasattr(self, "processor"):
            self.processor.save_pretrained(output_dir)
        # Do not call parent's save_model (prevents saving the full model)


class CustomImageFolderDataset(Dataset):
    """
    Custom dataset that wraps torchvision's ImageFolder and applies a processor.
    """

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            image, label = self.dataset[idx]
        except (OSError, IOError) as e:
            print(f"Error loading image: {self.dataset.imgs[idx][0]} - {e}")
            # Skip problematic image and move to next
            return self.__getitem__((idx + 1) % len(self.dataset))
        processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": processed_image.squeeze(), "labels": label}


class AdvancedClassifierHead(nn.Module):
    """
    Advanced classifier head with Squeeze-and-Excitation (SE) block and deep layers.
    """

    def __init__(self, hidden_size, dropout_rate=0.3):
        super(AdvancedClassifierHead, self).__init__()
        # Squeeze-and-Excitation module
        self.se = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, hidden_size),
            nn.Sigmoid()
        )
        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, cls_token_output):
        se_weights = self.se(cls_token_output)
        cls_token_output = cls_token_output * se_weights
        logits = self.classifier(cls_token_output)
        return logits


class AdvancedClassifierHead_CLIP(nn.Module):
    """
    Classifier head with channel attention for CLIP image features.
    """

    def __init__(self, input_dim=768, hidden_dim=512, dropout_rate=0.3):
        super(AdvancedClassifierHead_CLIP, self).__init__()
        # Channel-wise attention module
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        # Deep classifier layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1)  # Binary classification
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        x = x * attention_weights
        logits = self.classifier(x)
        return logits


class CombinedModel(nn.Module):
    """
    End-to-end model combining a ViT encoder and an advanced classifier head.
    Used for binary classification tasks.
    """

    def __init__(self, vit_model, classifier_head):
        super().__init__()
        self.vit = vit_model.eval()
        self.classifier_head = classifier_head
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

    def forward(self, pixel_values, labels=None):
        # Extract [CLS] token output from ViT
        vit_outputs = self.vit(pixel_values, output_hidden_states=True)
        cls_token_output = vit_outputs.hidden_states[-1][:, 0, :]  # [CLS] token
        logits = self.classifier_head(cls_token_output)
        loss = None
        if labels is not None:
            labels = labels.float()  # Ensure labels are float for BCE loss
            loss = self.loss_fn(logits.squeeze(), labels)
        return {"logits": logits, "loss": loss}

    def predict(self, pixel_values):
        vit_outputs = self.vit(pixel_values, output_hidden_states=True)
        cls_token_output = vit_outputs.hidden_states[-1][:, 0, :]
        logits = self.classifier_head(cls_token_output)
        predictions = torch.sigmoid(logits).squeeze().round().cpu().numpy()
        return predictions


class CLIPBinaryClassifier(nn.Module):
    """
    Binary classifier using frozen CLIP image features and an advanced classifier head.
    """

    def __init__(self, clip_model, classifier_head):
        super(CLIPBinaryClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier_head = classifier_head
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values, labels=None):
        with torch.no_grad():
            # Freeze the CLIP image encoder during training
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
        preds = torch.sigmoid(torch.tensor(logits)).squeeze().round().cpu().numpy()
        return preds
