"""Regenerate docs/data_dictionary.md and data/processed/data_dictionary.csv from config."""
from offshore_risk import load_config
from offshore_risk.data.dictionary import build

if __name__ == "__main__":
    df = build(load_config())
    print(f"data dictionary: {len(df)} parameters")
