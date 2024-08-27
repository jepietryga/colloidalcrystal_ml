import torch.utils
from torch.utils.data import Dataset
import torch
import torch.utils.data
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import h5py
import numpy as np
import pandas as pd
import glob
import h5py
import time
from tempfile import TemporaryDirectory
import os

from pathlib import Path
from torchvision.transforms import v2 as T
import torchvision
from torch import nn

from facet_ml.segmentation.segmenter import ImageSegmenter

LABEL_TO_INT = {"B": 0, "C": 1, "MC": 2, "I": 3, "PS": 4, "V": 1, "PC": 4}
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


def repeat_channels(x):
    return x.repeat(3, 1, 1)


class get_transform:

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


# def get_transform(train, im_size=256):
#     transforms = [
#         T.ToTensor(),
#         T.Lambda(lambda x: x.repeat(3, 1, 1)),
#         T.Resize((im_size, im_size)),
#     ]
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#         transforms.append(T.RandomVerticalFlip(0.5)),
#         transforms.append(
#             T.RandomRotation(
#                 90,
#             )
#         )
#     transforms.append(T.ToDtype(torch.float, scale=True))
#     transforms.append(T.ToPureTensor())
#     return T.Compose(transforms)


def crop_to_nonzero(image, padding=5):
    """
    Crop a NumPy array image to the bounding box of its non-zero regions.

    Args:
        image (np.array): The input image array.

    Returns:
        np.array: The cropped image array.
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


def load_colloidal_datasets(
    h5_file: str,
    csv_file: str,
    split_frac: float = 0.7,
    batch_size: int = 15,
    num_workers: int = 8,
    stratify: bool = False,
    mode: str = "no_blur",
) -> tuple[dict, dict]:
    """
    Provided an h5 and csv, prepare and return two dicts
    with keys "train" and "test".
    The first dict holds the Dataloaders, the second holds size information
    """
    loaded_h5s = [h5_file]
    loaded_df = pd.read_csv(csv_file)

    # Separate into test/train, train gets augmented
    dataset_train = ColloidalDataset(
        loaded_df,
        h5_total=loaded_h5s,
        transforms=get_transform(train=True, mode=mode)(),
    )
    dataset_test = dataset_train = ColloidalDataset(
        loaded_df,
        h5_total=loaded_h5s,
        transforms=get_transform(train=False, mode=mode)(),
    )

    # Separate now into Dataloader subsets of each
    if not stratify:
        indices = torch.randperm(len(dataset_test)).tolist()
        index_cut = int(len(indices) * split_frac)

        dataset_train = torch.utils.data.Subset(dataset_train, indices[:index_cut])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[index_cut:])
    else:
        from sklearn.model_selection import StratifiedKFold

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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


import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), DoubleConv(512, 256)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), DoubleConv(256, 128)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), DoubleConv(128, 64)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), DoubleConv(64, 32)
        )
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(
            torch.cat(
                [
                    x4,  # 512
                    F.interpolate(
                        x5, x4.size()[2:], mode="bilinear", align_corners=True
                    ),
                ],  # Total: 1024
                dim=1,
            )
        )
        x = self.up2(
            torch.cat(
                [
                    x3,
                    F.interpolate(
                        x, x3.size()[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x = self.up3(
            torch.cat(
                [
                    x2,
                    F.interpolate(
                        x, x2.size()[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x = self.up4(
            torch.cat(
                [
                    x1,
                    F.interpolate(
                        x, x1.size()[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        logits = self.outc(x)
        logits = F.adaptive_avg_pool2d(
            logits, 1
        )  # Global average pooling to convert to class scores
        logits = logits.view(logits.size(0), -1)  # Flatten to (batch_size, n_classes)

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
                    # inputs = torch.tensor(inputs)
                    # labels = torch.tensor(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

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
                if phase == "val" and epoch_acc > best_acc:
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


#
# Update this datset to be bale to made from
# (csv + h5) OR ImageSegmenter
#
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

        # a = np.zeros( (len(INT_TO_LABEL),1) )
        # a[LABEL_TO_INT[label]] = 1
        # return img, torch.tensor(a)
        return img, LABEL_TO_INT.get(label, 4)

        raise NotImplemented

    def __len__(self):
        return self.n_images

    @classmethod
    def from_h5(cls, csv_path, h5_path, transforms=None):
        df = pd.read_csv(csv_path)
        h5_path = h5_path
        return cls(df, h5, None, "h5")

        raise NotImplemented

    @classmethod
    def from_image_segmenter(
        cls,
        image_segmenter: ImageSegmenter,
    ):
        df = image_segmenter.df

        return cls(df, None, image_segmenter, mode="ImageSegmenter")
        raise NotImplemented
