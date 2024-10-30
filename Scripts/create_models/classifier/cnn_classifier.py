from facet_ml.classification import cnn
import torch
from torch import nn
import numpy as np
from torchvision.models import resnet152

if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Load data
    csv_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_27_Rachel-C_Processed.csv"
    h5_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_16_Rachel-C_Training.h5"
    dataloaders, datasizes = cnn.load_colloidal_datasets_h5(
        csv_file=csv_path,
        h5_file=h5_path,
        num_workers=4,
        batch_size=16,
        split_frac=0.7,
        stratify=True,
        three_class_mode=True,
    )
    check_input, check_output = next(iter(dataloaders["test"]))

    print(check_input.shape, check_output.shape)

    # Load model
    model = cnn.get_model(3, resnet152(weights="DEFAULT"))
    # model = cnn.UNet(3, 5)
    model.to(device=device)

    ## Get model trianing information
    criterion = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.01  # 0.0001
    )

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
    torch.save(model.state_dict(), "resnet152_trained_dict.pth")
    torch.save(model, "resnet152_trained.pth")

    ### DEBUGGING OUTPUTS

    cnn_results = []
    actual = []
    actual_from_row = []
    region_list = []
    adjusted_region_list = []
    scores = []

    for inputs, labels in dataloaders["test"]:

        actual.append(labels)

        # CNN application
        with torch.no_grad():

            cnn_output = model(inputs)
            cnn_results.append(int(np.argmax(cnn_output.cpu())))  # Move to CPU
        from sklearn.metrics import f1_score

        scores.append(f1_score(labels, cnn_output))

        # Clean up to prevent memory issues
        del inputs, cnn_output
        torch.cuda.empty_cache()
