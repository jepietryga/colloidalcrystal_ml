from facet_ml.classification import cnn

from facet_ml.classification import cnn
import torch
from torch import nn
import numpy as np
from torchvision.models import resnet152
from pathlib import Path

if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Load data
    data_folder = (
        Path(__file__).parent.parent.parent.parent / "ProcessedData" / "Coco_v5"
    )
    csv_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_27_Rachel-C_Processed.csv"
    h5_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_16_Rachel-C_Training.h5"
    dataloaders, datasizes = cnn.load_colloidal_datasets_coco(
        str(data_folder),
        num_workers=16,
        batch_size=2,
    )

    # Load model
    model = cnn.get_model(2, resnet152(weights="DEFAULT"))
    model = cnn.get_binary_model(2, resnet152(weights="DEFAULT"))
    model = cnn.UNet(3, 2)
    model.to(device=device)

    ## Get model trianing information
    criterion = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0001)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model, loss_dict, accuracy_dict = cnn.train_model(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        num_epochs=50,
        dataloaders=dataloaders,
        dataset_sizes=datasizes,
    )

    # Make quick plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(ncols=2)
    for ii, key in enumerate(loss_dict.keys()):
        # ax = axes[ii]
        x = np.array(range(len(loss_dict[key])))

        for ii, (label, dict_oi) in enumerate(
            [("Loss", loss_dict), ("Accuracy", accuracy_dict)]
        ):
            ax = axes[ii]
            y = np.array(dict_oi[key])

            ax.plot(x, y, label=f"{label}: {key}")
            ax.legend()

    fig.savefig("Model_performance.png")
    torch.save(model.state_dict(), "resnet152_trained.pth")
