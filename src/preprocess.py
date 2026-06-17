import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_features(input_path=None, output_path=None):
    if input_path is None:
        input_path = PROJECT_ROOT / "data" / "raw" / "nasa_power_weather.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "weather_features.csv"

    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    monthly = pd.DataFrame()
    monthly["t2m_max_mean"] = df["T2M_MAX"].resample("M").mean()
    monthly["t2m_min_mean"] = df["T2M_MIN"].resample("M").mean()
    monthly["rainfall_sum"] = df["PRECTOTCORR"].resample("M").sum()
    monthly["humidity_mean"] = df["RH2M"].resample("M").mean()
    monthly["solar_mean"] = df["ALLSKY_SFC_SW_DWN"].resample("M").mean()

    monthly["month"] = monthly.index.month
    monthly["year"] = monthly.index.year
    monthly = monthly.reset_index().rename(columns={"date": "month_end"})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return monthly

if __name__ == "__main__":
    out = build_features()
    print(out.head())