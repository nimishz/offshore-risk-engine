"""Regenerate the synthetic operating history in data/synthetic/ (fixed seed)."""
from offshore_risk import load_config
from offshore_risk.data import synthetic

if __name__ == "__main__":
    files = synthetic.generate(load_config())
    for name, df in files.items():
        print(f"{name:22s} {len(df):6d} rows")
