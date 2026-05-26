"""
Build data/nyc_ac_tracts.geojson by joining:
  - LACE 2023 AC data (Census/ASU)
  - ACS 2022 5-year socioeconomic data (via CensusReporter API, no key needed)
  - NYC census tract geometries (Census TIGER cartographic boundary)
"""

import io
import json
import time
import zipfile

import geopandas as gpd
import pandas as pd
import requests

NYC_COUNTIES = {
    "005": "Bronx",
    "047": "Brooklyn",
    "061": "Manhattan",
    "081": "Queens",
    "085": "Staten Island",
}
STATE = "36"

LACE_URL = "https://www2.census.gov/programs-surveys/demo/datasets/lace/2023/LACE_23_Tract.csv"
CARTOGRAPHIC_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_36_tract_500k.zip"
CENSUSREPORTER_URL = "https://api.censusreporter.org/1.0/data/show/latest"

# CensusReporter table IDs
ACS_TABLES = "B19013,B17001,B25003,B02001,B03002,B01001,B25010"
BATCH_SIZE = 50  # CensusReporter handles ~100 but 50 is reliable


def download_lace() -> pd.DataFrame:
    print("Downloading LACE 2023 AC data...")
    df = pd.read_csv(LACE_URL, dtype={"STATE": str, "COUNTY": str, "TRACT": str})
    nyc = df[(df["STATE"] == STATE) & df["COUNTY"].isin(NYC_COUNTIES)].copy()
    nyc["GEOID"] = nyc["STATE"] + nyc["COUNTY"] + nyc["TRACT"]

    def clean(col):
        s = pd.to_numeric(col, errors="coerce")
        s[s < 0] = None
        return s

    nyc["ac_pct"] = clean(nyc["AC_PE"])
    nyc["no_ac_pct"] = clean(nyc["NO_AC_PE"])
    nyc["ac_pct_moe"] = clean(nyc["AC_PM"])
    nyc["occupied_units"] = clean(nyc["HSE_OCC_E"])
    nyc["borough"] = nyc["COUNTY"].map(NYC_COUNTIES)
    print(f"  {len(nyc)} NYC tracts")
    return nyc[["GEOID", "NAME", "borough", "ac_pct", "no_ac_pct", "ac_pct_moe", "occupied_units"]]


def download_geometries() -> gpd.GeoDataFrame:
    print("Downloading tract geometries from Census TIGER...")
    resp = requests.get(CARTOGRAPHIC_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall("/tmp/cb_tract_ny")
    gdf = gpd.read_file("/tmp/cb_tract_ny")
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)
    nyc_geo = gdf[gdf["COUNTYFP"].isin(NYC_COUNTIES)].copy()
    nyc_geo = nyc_geo.to_crs("EPSG:4326")
    print(f"  {len(nyc_geo)} tract geometries")
    return nyc_geo[["GEOID", "geometry"]]


def fetch_acs_batch(geo_ids: list[str]) -> dict:
    """Fetch ACS data for a batch of tract geo IDs from CensusReporter."""
    params = {
        "table_ids": ACS_TABLES,
        "geo_ids": ",".join(geo_ids),
    }
    resp = requests.get(CENSUSREPORTER_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", {})


def download_acs(geoids: list[str]) -> pd.DataFrame:
    print(f"Fetching ACS data via CensusReporter for {len(geoids)} tracts...")
    cr_ids = ["14000US" + g for g in geoids]
    all_data = {}
    batches = [cr_ids[i : i + BATCH_SIZE] for i in range(0, len(cr_ids), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        if i % 5 == 0:
            print(f"  batch {i + 1}/{len(batches)}...")
        result = fetch_acs_batch(batch)
        all_data.update(result)
        time.sleep(0.2)

    rows = []
    for cr_id, tables in all_data.items():
        geoid = cr_id.replace("14000US", "")
        row = {"GEOID": geoid}

        def e(table: str, var: str):
            """Extract estimate value safely."""
            return tables.get(table, {}).get("estimate", {}).get(var)

        row["median_income"] = e("B19013", "B19013001")

        pov_denom = e("B17001", "B17001001") or 0
        pov_num = e("B17001", "B17001002") or 0
        row["poverty_rate"] = round(pov_num / pov_denom * 100, 1) if pov_denom else None

        ten_denom = e("B25003", "B25003001") or 0
        row["renter_rate"] = round((e("B25003", "B25003003") or 0) / ten_denom * 100, 1) if ten_denom else None

        tot_pop = e("B01001", "B01001001") or 0
        male65 = sum(
            e("B01001", f"B01001{n:03d}") or 0
            for n in range(20, 26)  # 020-025: male 65-66, 67-69, 70-74, 75-79, 80-84, 85+
        )
        fem65 = sum(
            e("B01001", f"B01001{n:03d}") or 0
            for n in range(44, 50)  # 044-049: female 65-66, ...
        )
        row["elderly_rate"] = round((male65 + fem65) / tot_pop * 100, 1) if tot_pop else None
        row["total_pop"] = tot_pop

        race_denom = e("B02001", "B02001001") or 0
        row["black_rate"] = round((e("B02001", "B02001003") or 0) / race_denom * 100, 1) if race_denom else None
        row["white_rate"] = round((e("B02001", "B02001002") or 0) / race_denom * 100, 1) if race_denom else None

        hisp_denom = e("B03002", "B03002001") or 0
        row["hispanic_rate"] = round((e("B03002", "B03002012") or 0) / hisp_denom * 100, 1) if hisp_denom else None

        row["avg_household_size"] = e("B25010", "B25010001")

        rows.append(row)

    print(f"  Got ACS data for {len(rows)} tracts")
    return pd.DataFrame(rows)


def main():
    lace = download_lace()
    geo = download_geometries()
    geoids = geo["GEOID"].tolist()
    acs = download_acs(geoids)

    merged = geo.merge(lace, on="GEOID", how="left")
    merged = merged.merge(acs, on="GEOID", how="left")

    out_path = "data/nyc_ac_tracts.geojson"
    merged.to_file(out_path, driver="GeoJSON")

    # Reduce coordinate precision + clean NaN
    with open(out_path) as f:
        gj = json.load(f)

    def round_coords(coords, prec=5):
        if coords and isinstance(coords[0], list):
            return [round_coords(c, prec) for c in coords]
        return [round(c, prec) for c in coords]

    for feat in gj["features"]:
        if feat["geometry"]:
            feat["geometry"]["coordinates"] = round_coords(feat["geometry"]["coordinates"])
        props = feat["properties"]
        for k, v in list(props.items()):
            if v != v:  # NaN
                props[k] = None

    with open(out_path, "w") as f:
        json.dump(gj, f, separators=(",", ":"))

    import os
    mb = os.path.getsize(out_path) / 1024 / 1024
    ac_vals = [f["properties"].get("ac_pct") for f in gj["features"]]
    ac_vals = [v for v in ac_vals if v is not None]
    print(f"\nDone! {out_path} — {mb:.1f} MB, {len(gj['features'])} features")
    if ac_vals:
        print(f"  AC% range: {min(ac_vals):.1f} – {max(ac_vals):.1f}")


if __name__ == "__main__":
    main()
