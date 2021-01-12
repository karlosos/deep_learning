import pandas as pd
import matplotlib.pyplot as plt

def optimizers_activations_comparison():
    # Porównanie wszystkich klasyfikatorów
    df = pd.read_csv("./experiments/experiment_1.csv")

    # Best models for 100 epochs
    print("Najlepsze")
    df = df.sort_values(by=['acc', 'loss'], ascending=False)
    print(df.to_latex(index=False))

    # Group by architecture
    print(df.groupby('architecture').mean().loc[:, ['loss', 'acc']].sort_values(by=['acc', 'loss'], ascending=False).to_latex())

    # Group by learning rate
    print(df.groupby('lr').mean().loc[:, ['loss', 'acc']].sort_values(by=['acc', 'loss'], ascending=False).to_latex())


def epochs_test_rate_comparison():
    # Wpływ liczby epok dokładność i loss function na zbiorze testowym
    df = pd.read_csv("./experiments/epochs_experiment_2_0.02.csv", index_col="epoch")
    print(df.loc[999])
    df.plot()
    plt.title("Architektura 2")
    plt.show()
    print(df)

    df = pd.read_csv("./experiments/epochs_experiment_3_0.02.csv", index_col="epoch")
    print(df.loc[999])
    df.plot()
    plt.title("Architektura 3")
    plt.show()
    print(df)


def model_metrics():
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    from conv import load_data

    trainX, trainY, testX, testY = load_data()

    model = load_model("models/conv_3_1000_0.02.h5")
    res = model.evaluate(testX, (testY))
    print(res)


def epochs_comparison():
    # Wpływ liczby epok na dokładność i loss function na zbiorze testowym
    df = pd.read_csv("./experiments/experiment_3.csv", index_col="epochs")
    df["acc"].plot()
    plt.title("Architektura 2 - dokładność na zbiorze testowym")
    plt.show()

    print(df[["acc", "loss"]].to_latex())


if __name__ == "__main__":
    # optimizers_activations_comparison()
    epochs_comparison()
    # model_metrics()
