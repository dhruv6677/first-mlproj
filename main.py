from src.data_ingest import Dataingestion
from src.datapreprocess import DataProcessing
from src.buildmodel import SimpleLinearRegression

if __name__ == "__main__":
    data_ingest = Dataingestion("./Data/Advertising-update.csv")
    X, y, df = data_ingest.get_x_y()
    df.to_csv("./Data/Advertising-update.csv", index=False)
    print(df)

    data_process = DataProcessing(df)
    outliers = data_process.identify_outliers_zscore(df["TV"])
    # print(outliers)
  # build the model
    lr_model = SimpleLinearRegression(X, y)
    model = lr_model.summary()
