"""
The two face recognition networks and their pre-processing.

NN1 (target of the attacks): Inception-ResNet-v1 (FaceNet) trained on
VGGFace2, from facenet-pytorch, used as an 8631-way classifier on
160x160 images normalised to [-1, 1].

NN2 (transferability): ResNet-50 trained on MS1M and fine-tuned on
VGGFace2 (Cao et al., 2018), 224x224 BGR input with mean subtraction.
"""

import pickle

import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from facenet_pytorch import InceptionResnetV1
from torch import nn
from torchvision import transforms

from advfr import config
from advfr.nets.resnet50_vggface2 import resnet50


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nn1(device=None):
    model = InceptionResnetV1(pretrained="vggface2").eval()
    model.classify = True  # return the 8631 identity logits instead of the embedding
    return model.to(device or get_device())


def load_nn2(weights_path=config.NN2_WEIGHTS, device=None):
    model = resnet50(num_classes=config.NUM_CLASSES)
    with open(weights_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")  # Caffe-converted weights stored as numpy arrays
    model.load_state_dict({k.replace("module.", ""): torch.from_numpy(v) for k, v in raw.items()})
    return model.eval().to(device or get_device())


# PIL image -> NN1 input
preprocess_nn1 = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# PIL image -> NN2 input. Nearest-neighbour resizing keeps the pixel values of
# the adversarial perturbation instead of averaging them out.
preprocess_nn2 = transforms.Compose([
    transforms.Resize(288, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),           # [0, 255]
    transforms.Lambda(lambda x: x[[2, 1, 0], :, :]),  # RGB -> BGR
    transforms.Normalize(mean=[91.4953, 103.8827, 131.0912], std=[1.0, 1.0, 1.0]),
])


def art_classifier(model, device=None):
    """
    ART wrapper of NN1. Attacks work on inputs in [0, 1], so the L-inf budget
    is measured in that space (epsilon = 0.1 is 10% of the range); the
    normalisation to [-1, 1] is applied by the wrapper.
    """
    device = device or get_device()
    return PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(3, config.IMAGE_SIZE, config.IMAGE_SIZE),
        nb_classes=config.NUM_CLASSES,
        preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


@torch.no_grad()
def predict(model, x, batch_size=64):
    """Arg-max class of a model on an already pre-processed tensor batch."""
    device = next(model.parameters()).device
    out = [model(x[i:i + batch_size].to(device)).argmax(1).cpu() for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()
