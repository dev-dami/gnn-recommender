import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def load_ratings(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df


def load_movies(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_date", "url",
               "unknown", "Action", "Adventure", "Animation", "Children",
               "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
               "Sci-Fi", "Thriller", "War", "Western"],
        engine="python",
    )
    return df


def filter_interactions(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(filtered) == len(df):
            break
        df = filtered
    return df


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)

    return df, user2idx, item2idx


def build_interaction_matrix(df: pd.DataFrame, num_users: int, num_items: int) -> sparse.csr_matrix:
    rows = df["user_idx"].values
    cols = df["item_idx"].values
    data = np.ones(len(df), dtype=np.float32)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return mat


def split_data(df: pd.DataFrame, test_ratio: float = 0.1, val_ratio: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    n_test = int(len(df) * test_ratio)
    n_val = int(len(df) * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def preprocess(data_dir: Path, processed_dir: str = "data/processed",
               min_interactions: int = 5, test_ratio: float = 0.1,
               val_ratio: float = 0.1, seed: int = 42) -> dict:
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    ratings = filter_interactions(ratings, min_interactions)
    ratings, user2idx, item2idx = remap_ids(ratings)

    num_users = len(user2idx)
    num_items = len(item2idx)

    train_df, val_df, test_df = split_data(ratings, test_ratio, val_ratio, seed)

    train_mat = build_interaction_matrix(train_df, num_users, num_items)

    meta = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": int(len(ratings)),
        "user2idx": {str(k): int(v) for k, v in user2idx.items()},
        "item2idx": {str(k): int(v) for k, v in item2idx.items()},
    }

    sparse.save_npz(out / "train_interaction.npz", train_mat)

    train_df[["user_idx", "item_idx"]].to_csv(out / "train.csv", index=False)
    val_df[["user_idx", "item_idx"]].to_csv(out / "val.csv", index=False)
    test_df[["user_idx", "item_idx"]].to_csv(out / "test.csv", index=False)

    movies.to_csv(out / "movies.csv", index=False)

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(ratings)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return meta
