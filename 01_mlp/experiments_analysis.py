import pandas as pd
import matplotlib.pyplot as plt

def optimizers_activations_comparison():
    # Porównanie wszystkich klasyfikatorów
    df = pd.read_csv("./experiments/experiment_1_100_epochs.csv")

    # Best models for 100 epochs
    df = df[df.epochs != 10]
    df = df.sort_values(by=['acc', 'loss'], ascending=False)
    print(df.iloc[0:10])

    print(df.groupby('optimizer').mean())
    print(df.groupby('lr').mean())
    print(df.groupby('activation').mean())


def epochs_lr_comparison():
    # Wpływ liczby epok na dokładność i loss function
    df = pd.read_csv("./experiments/epochs_experiment_0.01.csv", index_col="epoch")
    df.plot()
    plt.title("Learning rate = 0.01")
    plt.show()
    print(df)

    df = pd.read_csv("./experiments/epochs_experiment_0.10.csv", index_col="epoch")
    df.plot()
    plt.title("Learning rate = 0.10")
    plt.show()
    print(df)


if __name__ == "__main__":
    epochs_lr_comparison()
