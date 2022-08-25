from spader.data_source import DataContainer
from pathlib import Path

if __name__ == "__main__":
    for dataset in ["adult", "beers", "flights", "hospital", "movies", "rayyan", "restaurant", "tax"]:
        data_container = DataContainer.from_path("data/" + dataset)

        values = []
        for idx, row in data_container.dirty_df.iterrows():
            for col in data_container.dirty_df.columns:
                values.append(f"COL {col}\tVAL {data_container.dirty_df.loc[idx, col]}\t{int(data_container.groundtruth_df.loc[idx, col])}")

        Path(f"data/rotom/{dataset}").mkdir(parents=True, exist_ok=True)
        with open(f"data/rotom/{dataset}/test.txt", "w") as f:
            f.write("\n".join(values))