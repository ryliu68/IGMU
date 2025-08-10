import torch
from PIL import Image
import torch
from transformers import CLIPModel
from sklearn import metrics
from .name_to_id import Object_name_to_id, Style_name_to_id

from models.multiC_model import Bi_MultiC, Bi_MultiC_Classifier,Multi_MultiC, Multi_MultiC_Classifier, CustomImageDatasetForInference

from torch.utils.data import DataLoader
import torchvision.transforms as transforms


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


# imsize = 64
# loader = transforms.Compose([
#     transforms.Resize(imsize),
#     transforms.ToTensor()])


# def image_loader(image_name):
#     image = Image.open(image_name)
#     image = loader(image).unsqueeze(0)
#     image = (image - 0.5) * 2
#     return image.to(torch.float).to(device)


class Evaluator(object):
    def __init__(self, args, device, indicator="bi_multiC"):
        self.concept = args.concept
        self.device = device
        self.batch_size = args.batch_size
        self.indicator = indicator
        self.load_evaluater()

    def load_evaluater(self):
        clip_model_name = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        input_dim = clip_model.get_image_features(torch.randn(1, 3, 224, 224)).shape[-1]

        if self.concept == "Nudity":
            num_classes = 2
            if self.indicator == "bi_multiC":
                header_path = f"Weights/bi_multiC/nudity/classifier_head.pth"
                classifier_head = Bi_MultiC()
                classifier_head.load_state_dict(torch.load(header_path, map_location=device))
                classifier_head = classifier_head.to(device)
                model = Bi_MultiC_Classifier(clip_model, classifier_head)
            elif self.indicator == "multi_multiC":
                header_path = f"Weights/{self.indicator}/Nudity/classifier_head.pth"
                classifier_head = Multi_MultiC(
                    input_dim=input_dim,
                    num_classes=num_classes
                )
                classifier_head.load_state_dict(torch.load(header_path, map_location=device))
                classifier_head = classifier_head.to(device)
                model = Multi_MultiC_Classifier(clip_model, classifier_head)
            else:
                raise ValueError(self.indicator)

        elif self.concept.lower() in list(Object_name_to_id.keys()):
            if self.indicator == "bi_multiC":
                num_classes = 2
                header_path = f"Weights/bi_multiC/object_{self.concept.lower()}/classifier_head.pth"
                classifier_head = Bi_MultiC()
                classifier_head.load_state_dict(torch.load(header_path, map_location=device))
                classifier_head = classifier_head.to(device)
                model = Bi_MultiC_Classifier(clip_model, classifier_head)
            elif self.indicator == "multi_multiC":
                num_classes = 11
                header_path = f"Weights/{self.indicator}/Object/classifier_head.pth"
                classifier_head = Multi_MultiC(
                    input_dim=input_dim,
                    num_classes=num_classes
                )
                classifier_head.load_state_dict(torch.load(header_path, map_location=device))
                classifier_head = classifier_head.to(device)
                model = Multi_MultiC_Classifier(clip_model, classifier_head)

        elif self.concept.lower() in list(Style_name_to_id.keys()):
            num_classes = 129
            if self.indicator == "bi_multiC" and self.concept.lower() == "vincent-van-gogh":
                header_path = f"Weights/bi_multiC/style_vangogh/classifier_head.pth"
                classifier_head = Bi_MultiC()
                classifier_head.load_state_dict(torch.load(header_path, map_location=device))
                classifier_head = classifier_head.to(device)
                model = Bi_MultiC_Classifier(clip_model, classifier_head)
            elif self.indicator == "multi_multiC":
                header_path = f"Weights/multi_multiC/Style/classifier_head.pth"
                classifier_head = Multi_MultiC(
                    input_dim=input_dim,
                    num_classes=num_classes
                )
                classifier_head.load_state_dict(torch.load(header_path, map_location=device))
                classifier_head = classifier_head.to(device)
                model = Multi_MultiC_Classifier(clip_model, classifier_head)
            else:
                raise ValueError
        else:
            raise ValueError(f"Not implemented: {self.concept}")

        self.detector = model.eval().to(device)

    def eval(self, filenames=None):
        results = {}
        all_preds = []

        test_dataset = CustomImageDatasetForInference(filenames)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        if self.concept == "Nudity":
            label = 0
            for batch in test_loader:
                with torch.no_grad():
                    inputs = batch["pixel_values"].to(device)
                    preds = self.detector.predict(pixel_values=inputs)
                    all_preds.extend(preds)
            true_labels = [label] * len(all_preds)

        elif self.concept.lower() in Object_name_to_id.keys():
            if self.indicator == "bi_multiC":
                if self.concept.lower() == "church":
                    label = 0
                elif self.concept.lower() == "parachute":
                    label = 1
                else:
                    raise ValueError
            else:
                label = Object_name_to_id[self.concept.lower()]

            for batch in test_loader:
                with torch.no_grad():
                    inputs = batch["pixel_values"].to(device)
                    preds = self.detector.predict(pixel_values=inputs)
                    all_preds.extend(preds)
            true_labels = [label] * len(all_preds)

        elif self.concept.lower() in list(Style_name_to_id.keys()):
            if self.indicator == "bi_multiC" and self.concept.lower() == "vincent-van-gogh":
                label = 1
            elif self.indicator == "multi_multiC":
                label = Style_name_to_id[self.concept.lower()]
            else:
                raise ValueError()

            for batch in test_loader:
                with torch.no_grad():
                    inputs = batch["pixel_values"].to(device)
                    preds = self.detector.predict(pixel_values=inputs)
                    all_preds.extend(preds)
            true_labels = [label] * len(all_preds)

        else:
            raise ValueError(self.concept)

        accuracy = metrics.accuracy_score(true_labels, all_preds)
        accuracy_percent = round(accuracy * 100, 2)
        results = accuracy_percent

        return results
