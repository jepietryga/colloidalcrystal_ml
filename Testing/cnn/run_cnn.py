from facet_ml.classification.cnn import *


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from tempfile import TemporaryDirectory 

def train_model(model, criterion, optimizer, scheduler, num_epochs,
    dataloaders:dict,dataset_sizes:dict
    ):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

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
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    my_sum = torch.sum(preds == labels.data)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, ({running_corrects}/{dataset_sizes[phase]})')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

# load a model pre-trained on COCO
if __name__ == "__main__":
    model = torchvision.models.resnet50(weights="DEFAULT")
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 5

    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    import os

    def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.resnet50(weights="DEFAULT")

        n_features = model.fc.in_features
        
        model.fc = nn.Linear(n_features, num_classes)

        return model

    from torchvision.transforms import v2 as T
    import utils

    im_size = 256
    def get_transform(train):
        transforms = [
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3,1,1)),
            T.Resize((im_size,im_size))
        ]
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 5
    # use our dataset and defined transformations

    ## WINDOWS
    csv_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_27_Rachel-C_Processed.csv"
    h5_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_16_Rachel-C_Training.h5"

    ## MAC
    csv_path =  "/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/ProcessedData/Training_Data_20240216/2024_02_27_Rachel-C_Processed.csv"
    h5_path = "/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Training/2024_02_15_Jacob-P_Training/4 nM 1.h5"

    loaded_df = pd.read_csv(
            csv_path
          )
    import glob

    loaded_h5s = [h5py.File(h5_path, "r")]
    dataset = ColloidalDataset(
        loaded_df, h5_total=loaded_h5s, transforms=get_transform(train=True)
    )
    dataset_test = ColloidalDataset(
        loaded_df, h5_total=loaded_h5s, transforms=get_transform(train=False)
    )

    # split the dataset in train and test set
    print(len(dataset))
    print(np.shape(dataset[0][0]))
    indices = torch.randperm(len(dataset)).tolist()
    print(np.shape(indices))
    index_cut = int(len(indices)*.7)
    print(len(dataset))
    dataset = torch.utils.data.Subset(dataset, indices[:index_cut])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[index_cut:])
    print(len(dataset),len(dataset_test))
    dataset_sizes = {
        "train":len(dataset),
        "val":len(dataset_test)
    }
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=15, shuffle=True, num_workers=0, 
        # collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=15,
        shuffle=False,
        num_workers=0,
        # collate_fn=utils.collate_fn,
    )
    dataloaders = {
        "train":data_loader,
        "val":data_loader_test
    }

    # get the model using our helper function
    print(num_classes)
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it just for 2 epochs
    num_epochs = 5

    from engine import *

    train_model(model, criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,scheduler=lr_scheduler,num_epochs=25,
        dataloaders=dataloaders, dataset_sizes=dataset_sizes)
    # for epoch in range(num_epochs):
    #     # train for one epoch, printing every 10 iterations
    #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    #     # update the learning rate
    #     lr_scheduler.step()
    #     # evaluate on the test dataset
    #     evaluate(model, data_loader_test, device=device)

    print("That's it!")

    import matplotlib.pyplot as plt

    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    # WINDOWS
    img_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\Images\Training\39.5 hold 4.bmp"
    img_path = "/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Images/Diagnostic_Images/Model_check-diagnostic_images_best/L-2_nM-3_au10_mixing-T_oven-T_embed-SiO2_07.tif"
    import cv2

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = image[np.newaxis, :, :]
    image = torch.tensor(image)

    print(np.shape(image))
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        # x = x[:3, ...].to(device)
        x = x.to(device)
        print(x)
        predictions = model(
            [
                x,
            ]
        )
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
        torch.uint8
    )
    image = image[:3, ...]
    print(predictions)
    print(pred)
    print(image)
    pred_labels = [
        f"{INT_TO_LABEL[int(label)]}: {score:.3f}"
        for label, score in zip(pred["labels"], pred["scores"])
    ]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.6).squeeze(1)
    output_image = draw_segmentation_masks(
        output_image, masks, alpha=0.5, colors="blue"
    )

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig("TESTING.png")
