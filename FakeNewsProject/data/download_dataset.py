import csv
from pathlib import Path
from urllib.request import urlopen

DATA_URL = "https://raw.githubusercontent.com/rishabhmisra/News-Articles-Dataset/master/fake_or_real_news.csv"
DATA_PATH = Path(__file__).resolve().parent / "fake_or_real_news.csv"


def download_dataset(url: str = DATA_URL, target_path: Path = DATA_PATH) -> None:
    """Download a public fake news dataset into the data folder."""
    if target_path.exists():
        print(f"Dataset already exists at {target_path}")
        return

    print(f"Downloading dataset from {url}")
    with urlopen(url) as response:
        data = response.read().decode("utf-8")

    target_path.write_text(data, encoding="utf-8")
    print(f"Saved dataset to {target_path}")


if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as error:
        print("Dataset download failed.")
        print(error)
        print(
            "If the downloader cannot fetch the full dataset, use the built-in sample file in data/fake_or_real_news.csv"
        )
