import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./experiments/experiment_1_100_epochs.csv")

    # Best models for 100 epochs
    df = df[df.epochs != 10]
    df = df.sort_values(by=['acc', 'loss'], ascending=False)
    print(df.iloc[0:10])

    print(df.groupby('optimizer').mean())
    print(df.groupby('lr').mean())
    print(df.groupby('activation').mean())
