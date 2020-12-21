import pandas as pd
import matplotlib.pyplot as plt

def optimizers_activations_comparison():
    # Porównanie wszystkich klasyfikatorów
    df = pd.read_csv("./experiments/experiment_1_100_epochs.csv")

    # Best models for 100 epochs
    df = df[df.epochs != 10]
    print("Najlepsze")
    df = df.sort_values(by=['acc', 'loss'], ascending=False)
    print(df.iloc[0:10].to_latex(index=False))

    print("Najgorsze")
    df = df.sort_values(by=['acc', 'loss'], ascending=True)
    print(df.iloc[0:10].to_latex(index=False))

    print(df.groupby('optimizer').mean().loc[:, ['loss', 'acc']].sort_values(by=['acc', 'loss'], ascending=False).to_latex())
    print(df.groupby('lr').mean().loc[:, ['loss', 'acc']].sort_values(by=['acc', 'loss'], ascending=False).to_latex())
    print(df.groupby('activation').mean().loc[:, ['loss', 'acc']].sort_values(by=['acc', 'loss'], ascending=False).to_latex())


def epochs_lr_comparison():
    # Wpływ liczby epok na dokładność i loss function
    df = pd.read_csv("./experiments/epochs_experiment_0.01.csv", index_col="epoch")
    print(df.loc[999])
    df.plot()
    plt.title("Learning rate = 0.01")
    plt.show()
    print(df)

    df = pd.read_csv("./experiments/epochs_experiment_0.10.csv", index_col="epoch")
    print(df.loc[999])
    df.plot()
    plt.title("Learning rate = 0.10")
    plt.show()
    print(df)


if __name__ == "__main__":
    # optimizers_activations_comparison()
    epochs_lr_comparison()
