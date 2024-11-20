import torch.utils
from torch.utils.data import Dataset
import torch
import torch.utils.data
from torch import nn
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
# from torchvision.transforms.v2 import functional as F
import torch.nn.functional as F
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models import resnet152


from sklearn.model_selection import StratifiedKFold
import h5py
import numpy as np
import pandas as pd
import json
import glob
import h5py
import time
from tempfile import TemporaryDirectory
import os
import cv2
from pathlib import Path
import random

from PIL import Image

from facet_ml.segmentation.segmenter import ImageSegmenter

# Coco labels were slightly off to be more verbose
COCO_TO_LABEL = {"crystals": "C", "background": "B", "fused": "MC", "incomplete": "I"}

LABEL_TO_INT = {"B": 0, "C": 1, "MC": 2, "I": 3, "PS": 4, "V": 1, "PC": 4}
LABEL_TO_INT_3_CLASS = {"B": 2, "C": 0, "MC": 1, "I": 2, "PS": 2, "V": 0, "PC": 2}
INT_3_CLASS_TO_LABEL = {0: "C", 1: "MC", 2: "I"}
INT_TO_LABEL = {0: "B", 1: "C", 2: "MC", 3: "I", 4: "PS"}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_model(num_classes, model_choice=torchvision.models.resnet50(weights="DEFAULT")):
    """
    Simple helper fucntio nfor returning a model
    """
    # load an instance segmentation model pre-trained on COCO
    model = model_choice

    n_features = model.fc.in_features

    model.fc = nn.Linear(n_features, num_classes)

    return model


def get_binary_model(
    num_classes, model_choice=torchvision.models.resnet50(weights="DEFAULT")
):
    """
    Modify a ResNet model for binary segmentation (background vs foreground).
    """
    # Load the chosen ResNet model
    model = model_choice

    # Remove the final fully connected layer
    model.fc = nn.Identity()

    # Remove the average pooling layer
    model.avgpool = nn.Identity()

    # Add a segmentation head: 1x1 convolution to produce num_classes output channels
    model.segmentation_head = nn.Conv2d(2048, num_classes, kernel_size=1)

    return model


def repeat_channels(x):
    '''
    Given a 2D image, repeat its channel
    '''
    return x.repeat(3, 1, 1)


class get_transform:
    '''
    Transform class that can be pickled
    '''

    def __init__(self, train, im_size=256, mode: str = "no_blur"):
        self.train = train
        self.im_size = im_size
        self.mode = mode

        transforms = [
            T.ToTensor(),
            T.Lambda(repeat_channels),
            T.Resize((im_size, im_size)),
        ]
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomVerticalFlip(0.5)),
            transforms.append(
                T.RandomRotation(
                    90,
                )
            )
        if self.mode == "blur":
            transforms.append(T.GaussianBlur(7))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())

        self.transforms = transforms

    def __call__(self):
        return T.Compose(self.transforms)


def crop_to_nonzero(image, padding=5):
    """
    Crop a NumPy array image to the bounding box of its non-zero regions.

    Args:
        image (np.array): The input image array.
    """
    # Find the non-zero elements in the array
    non_zero_indices = np.nonzero(image)

    # Get the bounding box coordinates
    min_row, min_col = np.min(non_zero_indices[0]), np.min(non_zero_indices[1])
    max_row, max_col = np.max(non_zero_indices[0]), np.max(non_zero_indices[1])

    # Crop the image to the bounding box
    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1]

    # Pad image
    pad_image = np.pad(cropped_image, padding, mode="constant", constant_values=0)

    return pad_image


def load_colloidal_datasets_h5(
    h5_file: str,
    csv_file: str,
    split_frac: float = 0.7,
    batch_size: int = 15,
    num_workers: int = 8,
    stratify: bool = False,
    train_transforms=get_transform(train=True)(),
    test_transforms=get_transform(train=False)(),
    three_class_mode: bool = True,
) -> tuple[dict, dict]:
    """
    Provided an h5 and csv, prepare and return two dicts
    with keys "train" and "test".
    The first dict holds the Dataloaders, the second holds size information
    Args:
        h5_file (str) : Path to the h5_file of interest
        csv_file (str) : Paath to the csv_file of interest
        split_frac (float) : Fractional splitting of ddata in the Train/Test set
        batch_size (int) : Batch size for moddel loadding
        num_workers (int) : Torch workers
        stratify (bool) : Command to straify data splitting of the classes
        train_transforms (torch.Compose) : List of transforms onto the training data
        test_transforms (torch.Compose) : List of transforms onto the test data
    """
    loaded_h5s = [h5_file]
    loaded_df = pd.read_csv(csv_file)

    # Separate into test/train, train gets augmented
    dataset_train = ColloidalDataset(
        loaded_df,
        h5_total=loaded_h5s,
        transforms=train_transforms,
    )
    dataset_test = ColloidalDataset(
        loaded_df,
        h5_total=loaded_h5s,
        transforms=test_transforms,
    )

    # Separate now into Dataloader subsets of each
    if not stratify:
        indices = torch.randperm(len(dataset_test)).tolist()
        index_cut = int(len(indices) * split_frac)

        dataset_train = torch.utils.data.Subset(dataset_train, indices[:index_cut])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[index_cut:])
    else:


        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if three_class_mode:
            processed_df_labels = [
                LABEL_TO_INT_3_CLASS.get(row.label, 2)
                for _, row in dataset_train.df.iterrows()
            ]
        else:
            processed_df_labels = [
                LABEL_TO_INT.get(row.label, 4) for _, row in dataset_train.df.iterrows()
            ]
        (train_indices, test_indices) = next(
            kf.split(np.zeros(len(dataset_train)), processed_df_labels)
        )
        dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
        dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

    # Create the dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create dicts for return
    dataloaders = {"train": dataloader_train, "test": dataloader_test}
    datasize = {"train": len(dataset_train), "test": len(dataset_test)}

    return dataloaders, datasize

transform = T.Compose(
    [
        # T.ToPILImage(),  # Convert NumPy array (or tensor) to PIL Image
        T.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        T.RandomVerticalFlip(p=0.5),  # 50% chance of vertical flip
        T.RandomRotation(
            degrees=15
        ),  # Rotate the image randomly between -15 and +15 degrees
        # Unclear how these transforms impact the mask
        # T.ColorJitter(
        #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        # ),  # Random color jitter
        # T.RandomResizedCrop(
        #     size=(256, 256), scale=(0.8, 1.0)
        # ),  # Resize and randomly crop
        # T.GaussianBlur(
        #     kernel_size=3, sigma=(0.1, 2.0)
        # ),  # Apply Gaussian blur with random sigma
        T.ToTensor(),  # Convert PIL image back to tensor
    ]
    )
class CustomTransform:
    def __init__(self):
        self.transform = transform

    def __call__(self, img, mask):

        # img = T.ToPILImage()(img.to(torch.uint8))
        mask = T.ToPILImage()(mask.to(torch.uint8))

        seed = random.randint(0, 2**32)
        torch.manual_seed(seed)
        img = self.transform(img)
        
        torch.manual_seed(seed)
        mask = self.transform(mask)
        
        return img, mask

def load_colloidal_datasets_coco(
    parent_dir: str,
    training_dir: str = "train",
    testing_dir: str = "test",
    num_workers: int = 8,
    batch_size: int = 2,
    mark_edges: bool = False
):
    """
    Use the COCO data to train CNN background-foreground pixel classifier
    """

    # Heavy augmentation is needed
    

    dataset_train = CocoColloidalDataset(
        root=os.path.join(parent_dir, training_dir),
        annotation_file=os.path.join(
            parent_dir, training_dir, "_annotations.coco.json"
        ),
        transforms=CustomTransform(),
        mark_edges=mark_edges
    )
    dataset_test = CocoColloidalDataset(
        root=os.path.join(parent_dir, testing_dir),
        annotation_file=os.path.join(parent_dir, testing_dir, "_annotations.coco.json"),
        transforms=CustomTransform(),
        mark_edges=mark_edges
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataloaders = {"train": dataloader_train, "test": dataloader_test}

    datasize = {
        "train": len(dataset_train),
        "test": len(dataset_test),
    }
    return dataloaders, datasize

## Write model variations here

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),  # Dropout layer added
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 features:list = [64, 128, 256, 512],
                 dim=256):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes  
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Build downs
        pre_feat = n_channels
        for feature in features:
            self.downs.append(SegDoubleConv(pre_feat,feature) )
            pre_feat = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                ))
            self.ups.append(SegDoubleConv(feature*2,feature))

        self.bottom = SegDoubleConv(features[-1],features[-1]*2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((dim, dim))
        self.outc = nn.Conv2d(features[0], self.n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom
        x = self.bottom(x)
        skip_connections = skip_connections[::-1]

        # Up range(len(self.ups)):#
        for ii in range(0, len(self.ups), 2):

            x = self.ups[ii](x)
            skip_connection = skip_connections[ii // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[ii+1](concat_skip)
        
        logits = self.outc(x)
        logits = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=True)

        logits = F.adaptive_avg_pool2d(
            logits, 1
        )  # Global average pooling to convert to class scores
        logits = logits.view(logits.size(0), -1)  # Flatten to (batch_size, n_classes)

        return logits


### U-Net for semantic segmentation ###
class SegDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SegUNet(nn.Module):
    '''
    Functionally, this is identical to UNet but changes to do pixel classification instead of
    region classification.
    '''
    def __init__(self, n_channels, 
                 n_classes=2, 
                 dim=256,
                 features:list = [64, 128, 256, 512]
                 ):
        super(SegUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes  # Binary
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Build downs
        pre_feat = n_channels
        for feature in features:
            self.downs.append(SegDoubleConv(pre_feat,feature) )
            pre_feat = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                ))
            self.ups.append(SegDoubleConv(feature*2,feature))

        self.bottom = SegDoubleConv(features[-1],features[-1]*2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((dim, dim))
        self.outc = nn.Conv2d(features[0], self.n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom
        x = self.bottom(x)
        skip_connections = skip_connections[::-1]

        # Up range(len(self.ups)):#
        for ii in range(0, len(self.ups), 2):
            x = self.ups[ii](x)
            skip_connection = skip_connections[ii // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[ii+1](concat_skip)
        
        logits = self.outc(x)

        logits = self.outc(x)
        logits = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=True)

        # # logits = F.adaptive_avg_pool2d(
        # #     logits, 1
        # # )  # Global average pooling to convert to class scores
        # # logits = logits.view(logits.size(0), -1)  # Flatten to (batch_size, n_classes)


        # Apply softmax activation to get class probabilities
        return logits

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders: dict,
    dataset_sizes: dict,
):
    """
    Train num_epochs
    """
    since = time.time()

    # Create a temporary directory to save training checkpoints
    loss_dict = {"train": [], "test": []}
    accuracy_dict = {"train": [], "test": []}
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs,1)
                        preds = torch.argmax(outputs,1)
                        if isinstance(model,UNet) or isinstance(model,type(resnet152())):
                            loss = criterion(outputs,labels)
                        elif isinstance(model,SegUNet):
                            labels = labels.squeeze(1)
                            loss = criterion(outputs, labels)
                        else:
                            raise Exception(f"{type(model)} not supported in this function")
                        # exit()

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / dataset_sizes[phase]).cpu()

                loss_dict[phase].append(epoch_loss)
                accuracy_dict[phase].append(epoch_acc)

                print(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, ({running_corrects}/{dataset_sizes[phase]})"
                )

                # deep copy the model
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, loss_dict, accuracy_dict


def train_model_coco(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders: dict,
    dataset_sizes: dict,
):
    """
    Train num_epochs
    """
    since = time.time()

    # Create a temporary directory to save training checkpoints
    loss_dict = {"train": [], "test": []}
    accuracy_dict = {"train": [], "test": []}
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)

                    labels = labels.to(device).unsqueeze(1).float()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / dataset_sizes[phase]).cpu()

                loss_dict[phase].append(epoch_loss)
                accuracy_dict[phase].append(epoch_acc)

                print(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, ({running_corrects}/{dataset_sizes[phase]})"
                )

                # deep copy the model
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, loss_dict, accuracy_dict


def load_model(model_config_pth, model_class, num_classes=5):
    """
    Given a model path, load it in
    """
    model = model_class()
    n_features = model.fc.in_features

    model.fc = nn.Linear(n_features, num_classes)
    model.load_state_dict(torch.load(model_config_pth))
    model.eval()

    return model

class ColloidalDataset(Dataset):

    def __init__(
        self,
        df_total,
        h5_total: list = None,
        image_segmenter: ImageSegmenter = None,
        mode: str = "h5",
        transforms=None,
    ):
        """
        Given the dataframe of each file,
        associate the data rows to their binary masks in the h5s.
        """
        if (h5_total is None) and (image_segmenter is None):
            raise Exception("Need at least one source of image data")
        self.transforms = transforms
        self.mode = mode

        # Load the row dataframe, organize it by filename
        self.df: pd.DataFrame = df_total
        self.df.reset_index(drop=True, inplace=True)
        self.df.sort_values(by="Filename", inplace=True)
        self.filenames = [Path(fn).stem for fn in self.df.Filename.unique()]
        self.n_images = len(self.df)

        ## If h5 mode, use these variables
        # Load the h5s, then load the masks associated
        if not isinstance(h5_total, list):
            h5_files = [h5_total]
        else:
            h5_files = h5_total

        self._h5_files = h5_files

        ## If ImageSegmenter Mode, use these variables
        self._image_segmenter = image_segmenter

    def _get_h5_img(self, label, filename, region):
        # From the h5s, grab the h5 associated with the image name
        h5_name = filename
        h5_file = None
        for h5 in self._h5_files:
            data = h5py.File(h5, "r")

            if h5_name in data.keys():
                h5_file = data
                break
            else:
                data.close()
        if h5_file is None:
            raise Exception(f"'{h5_name}' does not exist in provided h5 files")

        # Grab the h5 regions
        data_group = h5_file[h5_name]
        region_index = region - 1

        img = data_group["Regions"][region_index, :, :]

        # Apply bounding
        img = crop_to_nonzero(img)
        return img

    def __getitem__(self, idx):

        # Get Row of data
        row = self.df.iloc[idx]

        # For row of the data get its label, file name, and region
        label = row.Labels
        filename = str(Path(row.Filename).stem)
        region = row.Region

        if self.mode == "h5":
            img = self._get_h5_img(label, filename, region)
        elif self.mode == "ImageSegmenter":
            img_oi = self._image_segmenter.image_cropped
            img = self._image_segmenter._grab_region(img_oi, region, 0, 5)

        # target = temp_df.to_dict()

        if self.transforms is not None:
            img = self.transforms(img)

        label_val = LABEL_TO_INT_3_CLASS.get(label, 2)
        return img, label_val

    def __len__(self):
        return self.n_images

    @classmethod
    def from_h5(cls, csv_path, h5_path, transforms=None):
        df = pd.read_csv(csv_path)
        h5_path = h5_path
        return cls(df, h5_path, None, "h5")

        raise NotImplemented

    @classmethod
    def from_image_segmenter(
        cls,
        image_segmenter: ImageSegmenter,
    ):
        df = image_segmenter.df

        return cls(df, None, image_segmenter, mode="ImageSegmenter")
        raise NotImplemented

class CocoColloidalDataset(Dataset):
    """
    This dataset is intended to be used w/ Coco labeled data
    """

    def __init__(self, root, annotation_file, patch_size=(256, 256), transforms=None,mark_edges=False):
        """
        Args:
            root (string): Root directory where images are stored.
            annotation_file (string): Path to the COCO annotations file.
            transforms (callable, optional): A function/transform that takes in
                                             an image and returns a transformed version.
        """
        self.root = root
        self.transforms = transforms
        self.patch_size = patch_size
        self.annotation_file = annotation_file
        self.mark_edges = mark_edges

        # Load annotations
        with open(annotation_file, "r") as f:
            self.coco = json.load(f)

        # Get all images and annotations
        self.images = self.coco["images"]
        self.annotations = {ann["image_id"]: [] for ann in self.coco["annotations"]}
        for ann in self.coco["annotations"]:
            self.annotations[ann["image_id"]].append(ann)
        # self.categories = {option["id"]: option["name"] for option in ann["categories"]}

        # For each image, patchify to get the masks and data
        self.patches, self.masks = self._patchify_data()

    def _patchify_data(self, y_factor: int = 4, x_factor: int = 8):
        """
        Split each image and its respective annotations
        """
        total_image_patches = []
        total_mask_patches = []
        for idx, image in enumerate(self.images):
            # Get the image and annotations
            img_info = self.images[idx]
            img_id = img_info["id"]
            img_path = os.path.join(self.root, img_info["file_name"])

            # Load image with cv2
            image = cv2.imread(img_path)
            image = (
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8) / 255
            )  # Convert from BGR to RGB
            # image = image.transpose((2, 0, 1))  # Shape: (3, 1024, 1280)

            # Get the annotations for the current image
            annotations = self.annotations[img_id]

            boxes = []
            labels = []
            for ann in annotations:
                bbox = ann["bbox"]
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = xmin + bbox[2]
                ymax = ymin + bbox[3]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann["category_id"])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # For each annotation, convert to background and not background and get a mask

            mask = np.zeros(image.shape[:2], np.int16)
            empty = np.zeros(image.shape[:2], np.int16)
            edges = np.zeros(image.shape[:2], np.int16)
            for ii, ann in enumerate(annotations):
                
                fill_val = int(
                    ann["category_id"] != 2
                )  # 4 should be background, but make this more explicit later
                polygon = np.array(ann["segmentation"], dtype=np.int32).reshape(-1, 2)
                mask = cv2.fillPoly(mask, [polygon], fill_val)

                if self.mark_edges and fill_val:

                    temp_poly = cv2.fillPoly(empty, [polygon], fill_val)
                    expand_poly = cv2.dilate(temp_poly, np.ones((5, 5)), iterations=1)
                    shrink_poly = cv2.erode(temp_poly, np.ones((5, 5)), iterations=1)
                    edge = cv2.subtract(expand_poly, shrink_poly)
                    edges = cv2.add(edges, edge)
            if self.mark_edges:
                mask[edges > 1] = 2
            imgs = []
            masks = []
            if self.patch_size is not None:
                for y in range(
                    0, image.shape[0] - self.patch_size[0] + 1, self.patch_size[0]
                ):
                    for x in range(
                        0, image.shape[1] - self.patch_size[1] + 1, self.patch_size[1]
                    ):
                        img_patch = image[
                            y : y + self.patch_size[0], x : x + self.patch_size[1], :
                        ]
                        mask_patch = mask[
                            y : y + self.patch_size[0], x : x + self.patch_size[1]
                        ]

                        imgs.append(img_patch)
                        masks.append(mask_patch)

                total_image_patches.extend(imgs)
                total_mask_patches.extend(masks)
            else:
                total_image_patches.append(image)
                total_mask_patches.append(mask)
        return total_image_patches, total_mask_patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]
        mask = self.masks[idx]

        if img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Convert NumPy array to PIL Image
        img = Image.fromarray(img)
        # img = torch.tensor(img)

        # Convert mask to tensor
        # mask = Image.fromarray(mask,"L") #torch.tensor(mask, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        if self.transforms:
            img, mask = self.transforms(img, mask)
        mask = torch.ceil(mask*255)
        mask = mask.long()

        return img, mask
