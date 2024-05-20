from facet_ml.classification.mask_rcnn import *


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 5
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    import os

    def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    from torchvision.transforms import v2 as T
    import utils

    def get_transform(train):
        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 5
    # use our dataset and defined transformations
    loaded_df = pd.read_csv(
        r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_27_Rachel-C_Processed.csv"
    )
    import glob

    h5_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_16_Rachel-C_Training.h5"
    loaded_h5s = [h5py.File(h5_path, "r")]
    dataset = ColloidalDataset(
        loaded_df, h5_total=loaded_h5s, transforms=get_transform(train=True)
    )
    dataset_test = ColloidalDataset(
        loaded_df, h5_total=loaded_h5s, transforms=get_transform(train=False)
    )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    print(np.shape(indices))
    dataset = torch.utils.data.Subset(dataset, indices[:-3])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-3:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

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
    num_epochs = 60

    from engine import *

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    import matplotlib.pyplot as plt

    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    img_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\Images\Training\39.5 hold 4.bmp"
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
