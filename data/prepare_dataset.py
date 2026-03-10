from pathlib import Path
import pandas as pd


def infer_label_and_metadata(file_path: Path) -> tuple[int, str, str]:
    """
    Infer:
    - label: 1 = deceptive, 0 = truthful
    - source: folder/source information
    - polarity: positive / negative / unknown
    """
    path_str = str(file_path).lower()

    label = 1 if "deceptive" in path_str else 0
    polarity = "positive" if "positive_polarity" in path_str else "negative" if "negative_polarity" in path_str else "unknown"

    if "tripadvisor" in path_str:
        source = "tripadvisor"
    elif "mturk" in path_str:
        source = "mturk"
    else:
        source = "unknown"

    return label, source, polarity


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw" / "op_spam_v1.4"
    output_csv = project_root / "data" / "reviews.csv"

    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found at: {raw_data_dir}\n"
            "Please place the extracted op_spam_v1.4 folder inside data/raw/."
        )

    rows = []

    txt_files = list(raw_data_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under: {raw_data_dir}")

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            continue

        label, source, polarity = infer_label_and_metadata(file_path)

        rows.append(
            {
                "text": text,
                "label": label,
                "source": source,
                "polarity": polarity,
                "file_path": str(file_path.relative_to(project_root)),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid review rows were created.")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Created dataset: {output_csv}")
    print(f"Total rows: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nPolarity distribution:")
    print(df["polarity"].value_counts())
    print("\nSource distribution:")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()