from facet_ml.classification import cnn
import torch
from torch import nn
import numpy as np
from torchvision.models import resnet152
import itertools
import matplotlib.pyplot as plt
import pandas as pd


def train_and_evaluate(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    dataloaders,
    dataset_sizes,
    device,
):
    model.to(device=device)
    model, loss_dict, accuracy_dict = cnn.train_model(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        num_epochs=num_epochs,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
    )
    return loss_dict, accuracy_dict


if __name__ == "__main__":
    # Train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Load data
    csv_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_27_Rachel-C_Processed.csv"
    h5_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\ProcessedData\Training_Data_20240216\2024_02_16_Rachel-C_Training.h5"
    dataloaders, datasizes = cnn.load_colloidal_datasets(
        csv_file=csv_path, h5_file=h5_path, num_workers=2, batch_size=16, split_frac=0.7
    )

    # Define hyperparameters to tune
    learning_rates = [0.001, 0.005, 0.01]
    momentums = [0.9, 0.95]
    weight_decays = [0.0001, 0.0005]
    step_sizes = [3, 5]
    gammas = [0.1, 0.5]

    best_accuracy = 0.0
    best_hyperparams = None
    best_loss_dict = None
    best_accuracy_dict = None

    results = []

    # Iterate over all combinations of hyperparameters
    for lr, momentum, weight_decay, step_size, gamma in itertools.product(
        learning_rates, momentums, weight_decays, step_sizes, gammas
    ):
        print(
            f"Training with lr={lr}, momentum={momentum}, weight_decay={weight_decay}, step_size={step_size}, gamma={gamma}"
        )

        # Load model
        model = cnn.get_model(5, resnet152(weights="DEFAULT"))

        # Define criterion, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        # Train and evaluate the model
        loss_dict, accuracy_dict = train_and_evaluate(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            num_epochs=20,
            dataloaders=dataloaders,
            dataset_sizes=datasizes,
            device=device,
        )

        # Track results
        val_accuracy = accuracy_dict["test"][-1]
        val_loss = loss_dict["test"][-1]
        results.append(
            {
                "learning_rate": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "step_size": step_size,
                "gamma": gamma,
                "validation_accuracy": val_accuracy,
                "validation_loss": val_loss,
            }
        )

        # Check if this combination is the best so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_hyperparams = (lr, momentum, weight_decay, step_size, gamma)
            best_loss_dict = loss_dict
            best_accuracy_dict = accuracy_dict

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    print("Results saved to hyperparameter_tuning_results.csv")

    print(
        f"Best hyperparameters: lr={best_hyperparams[0]}, momentum={best_hyperparams[1]}, weight_decay={best_hyperparams[2]}, step_size={best_hyperparams[3]}, gamma={best_hyperparams[4]}"
    )
    print(f"Best validation accuracy: {best_accuracy}")

    # Plot the best model performance
    fig, axes = plt.subplots(ncols=2)
    for ii, key in enumerate(best_loss_dict.keys()):
        x = np.array(range(len(best_loss_dict[key])))
        for ii, (label, dict_oi) in enumerate(
            [("Loss", best_loss_dict), ("Accuracy", best_accuracy_dict)]
        ):
            ax = axes[ii]
            y = np.array(dict_oi[key])
            ax.plot(x, y, label=f"{label}: {key}")
            ax.legend()

    fig.savefig("Best_Model_performance.png")
