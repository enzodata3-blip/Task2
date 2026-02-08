#!/usr/bin/env python3
"""
Boston Poverty Mapping — Tableau-Style Dashboard
Python replication of: https://github.com/susanli2016/Data-Analysis-with-R/blob/master/Mapping-Boston-Poverty.Rmd

Data: US Census ACS 2014 Five-Year Estimates, Table B17021
      (Poverty Status of Individuals in the Past 12 Months by Living Arrangement)
Geography: Suffolk County, Massachusetts (Boston)

R equivalents:
  acs14lite::acs14()     -> requests to Census API
  tigris::tracts()       -> geopandas from Census TIGER shapefile
  tigris::geo_join()     -> gdf.merge()
  ggplot2 choropleth     -> plotly Choroplethmapbox
  ggmap base layer       -> carto-positron tile layer

Dependencies:
  pip install requests pandas geopandas plotly numpy
"""

import requests
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np
import json

# ============================================================
# CONFIGURATION
# ============================================================
# Register a free Census API key at: https://api.census.gov/data/key_signup.html
# DEMO_KEY works but has rate limits (500 requests/day per IP).
CENSUS_API_KEY = ""  # Optional — register free at https://api.census.gov/data/key_signup.html
STATE_FIPS = "25"    # Massachusetts
COUNTY_FIPS = "025"  # Suffolk County (Boston)
OUTPUT_FILE = "boston_poverty_dashboard.html"


# ============================================================
# STEP 1: FETCH CENSUS ACS DATA  (R: acs14lite::acs14)
# ============================================================
def fetch_census_poverty_data():
    """
    Pull B17021 poverty estimates from ACS 2014 5-year API for all
    census tracts in Suffolk County.

    B17021_001E: Population for whom poverty status is determined (estimate)
    B17021_001M: Same, margin of error
    B17021_002E: Population below poverty level (estimate)
    B17021_002M: Same, margin of error
    """
    key_param = f"&key={CENSUS_API_KEY}" if CENSUS_API_KEY else ""
    url = (
        "https://api.census.gov/data/2014/acs/acs5"
        "?get=NAME,B17021_001E,B17021_002E,B17021_001M,B17021_002M"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}%20county:{COUNTY_FIPS}"
        f"{key_param}"
    )
    print("Fetching Census ACS data...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])

    numeric_cols = ["B17021_001E", "B17021_002E", "B17021_001M", "B17021_002M"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # R equivalent: 100 * (B17021_002E / B17021_001E)
    df["poverty_pct"] = 100 * (df["B17021_002E"] / df["B17021_001E"])
    df["poverty_moe"] = 100 * (df["B17021_002M"] / df["B17021_001E"])

    # Build GEOID for spatial join (state + county + tract)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["tract_label"] = df["NAME"].str.split(",").str[0]

    print(f"  {len(df)} tracts retrieved")
    return df


# ============================================================
# STEP 2: FETCH TIGER SHAPEFILE  (R: tigris::tracts)
# ============================================================
def fetch_tiger_shapefile():
    """
    Download 2014 Massachusetts TIGER census tract boundaries,
    filter to Suffolk County, reproject to WGS84 (EPSG:4326).
    """
    url = f"https://www2.census.gov/geo/tiger/TIGER2014/TRACT/tl_2014_{STATE_FIPS}_tract.zip"
    print("Downloading TIGER shapefile (Massachusetts tracts)...")
    gdf = gpd.read_file(url)
    gdf = gdf[gdf["COUNTYFP"] == COUNTY_FIPS].copy()
    gdf = gdf.to_crs(epsg=4326)
    print(f"  {len(gdf)} tracts in Suffolk County")
    return gdf


# ============================================================
# STEP 3: SPATIAL JOIN  (R: tigris::geo_join)
# ============================================================
def merge_spatial_data(census_df, gdf):
    """Join census poverty estimates onto TIGER tract boundaries by GEOID."""
    merged = gdf.merge(census_df, on="GEOID", how="left")
    merged = merged.dropna(subset=["poverty_pct"])
    print(f"  {len(merged)} tracts after join")
    return merged


# ============================================================
# STEP 4: STATISTICS
# ============================================================
def compute_stats(gdf):
    return {
        "mean":         gdf["poverty_pct"].mean(),
        "median":       gdf["poverty_pct"].median(),
        "max":          gdf["poverty_pct"].max(),
        "min":          gdf["poverty_pct"].min(),
        "std":          gdf["poverty_pct"].std(),
        "total_pop":    int(gdf["B17021_001E"].sum()),
        "total_poor":   int(gdf["B17021_002E"].sum()),
        "n_tracts":     len(gdf),
    }


# ============================================================
# STEP 5: BUILD TABLEAU-STYLE DASHBOARD
# ============================================================
def build_chart_map(gdf, geojson):
    """Choropleth map — equivalent of ggmap() poverty overlay."""
    blue_scale = [
        [0.00, "#f7fbff"],
        [0.20, "#deebf7"],
        [0.40, "#9ecae1"],
        [0.60, "#3182bd"],
        [0.80, "#08519c"],
        [1.00, "#08306b"],
    ]
    customdata = np.column_stack([
        gdf["tract_label"].fillna("").values,
        gdf["B17021_001E"].fillna(0).values,
        gdf["B17021_002E"].fillna(0).values,
        gdf["poverty_moe"].fillna(0).values,
    ])
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=list(range(len(gdf))),
        z=gdf["poverty_pct"].values,
        colorscale=blue_scale,
        zmin=0,
        zmax=gdf["poverty_pct"].quantile(0.95),
        marker_opacity=0.82,
        marker_line_width=0.8,
        marker_line_color="white",
        colorbar=dict(title="Poverty %", thickness=14, len=0.65,
                      tickformat=".0f", ticksuffix="%"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Poverty rate: <b>%{z:.1f}%</b><br>"
            "Population: %{customdata[1]:,.0f}<br>"
            "In poverty: %{customdata[2]:,.0f}<br>"
            "Margin of error: ±%{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=42.355, lon=-71.06), zoom=11),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="Poverty Rate by Census Tract", font=dict(size=14, color="#2c3e50"), x=0.5),
        height=500,
        paper_bgcolor="white",
    )
    return fig


def build_chart_histogram(gdf, stats):
    """Distribution of tract-level poverty rates with mean/median reference lines."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=gdf["poverty_pct"],
        nbinsx=18,
        marker_color="#4e79a7",
        marker_line_color="white",
        marker_line_width=1.2,
        opacity=0.9,
        hovertemplate="Range: %{x:.1f}%<br>Tracts: %{y}<extra></extra>",
    ))
    fig.add_vline(x=stats["mean"], line_dash="dash", line_color="#e15759", line_width=2,
                  annotation_text=f"Mean {stats['mean']:.1f}%",
                  annotation_font_color="#e15759", annotation_position="top right")
    fig.add_vline(x=stats["median"], line_dash="dot", line_color="#76b7b2", line_width=2,
                  annotation_text=f"Median {stats['median']:.1f}%",
                  annotation_font_color="#76b7b2", annotation_position="top left")
    fig.update_layout(
        title=dict(text="Distribution of Tract Poverty Rates", font=dict(size=14, color="#2c3e50"), x=0.5),
        xaxis_title="Poverty Rate (%)",
        yaxis_title="Number of Tracts",
        bargap=0.05,
        height=350,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#2c3e50"),
    )
    return fig


def build_chart_top15(gdf):
    """Horizontal bar chart of 15 highest-poverty tracts."""
    top15 = gdf.nlargest(15, "poverty_pct").sort_values("poverty_pct")
    fig = go.Figure(go.Bar(
        x=top15["poverty_pct"],
        y=top15["tract_label"],
        orientation="h",
        marker=dict(color=top15["poverty_pct"], colorscale="Blues", showscale=False),
        text=top15["poverty_pct"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Poverty: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top 15 Tracts by Poverty Rate", font=dict(size=14, color="#2c3e50"), x=0.5),
        xaxis_title="Poverty Rate (%)",
        xaxis=dict(range=[0, top15["poverty_pct"].max() * 1.15], showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(tickfont=dict(size=10)),
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        margin=dict(l=10, r=60),
        font=dict(color="#2c3e50"),
    )
    return fig


def build_chart_scatter(gdf):
    """Poverty rate vs. tract population with OLS trend line."""
    valid = gdf.dropna(subset=["B17021_001E", "poverty_pct"])
    z = np.polyfit(valid["B17021_001E"], valid["poverty_pct"], 1)
    x_line = np.linspace(valid["B17021_001E"].min(), valid["B17021_001E"].max(), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid["B17021_001E"],
        y=valid["poverty_pct"],
        mode="markers",
        marker=dict(size=8, color=valid["poverty_pct"], colorscale="Blues",
                    showscale=False, opacity=0.78, line=dict(width=0.5, color="white")),
        text=valid["tract_label"],
        hovertemplate="<b>%{text}</b><br>Population: %{x:,.0f}<br>Poverty: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=np.polyval(z, x_line),
        mode="lines", line=dict(color="#e15759", dash="dash", width=2),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text="Poverty Rate vs. Tract Population", font=dict(size=14, color="#2c3e50"), x=0.5),
        xaxis_title="Total Population",
        yaxis_title="Poverty Rate (%)",
        height=350,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        showlegend=False,
        font=dict(color="#2c3e50"),
    )
    return fig


def assemble_dashboard(gdf, stats):
    """Combine all charts into a single Tableau-style HTML dashboard."""
    geojson = json.loads(gdf.to_json())
    for i, f in enumerate(geojson["features"]):
        f["id"] = i

    fig_map = build_chart_map(gdf, geojson)
    fig_hist = build_chart_histogram(gdf, stats)
    fig_top15 = build_chart_top15(gdf)
    fig_scatter = build_chart_scatter(gdf)

    overall_rate = 100 * stats["total_poor"] / stats["total_pop"]

    map_html = fig_map.to_html(full_html=False, include_plotlyjs=False)
    hist_html = fig_hist.to_html(full_html=False, include_plotlyjs=False)
    top15_html = fig_top15.to_html(full_html=False, include_plotlyjs=False)
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Boston Poverty Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #f0f2f5; font-family: 'Segoe UI', Arial, sans-serif; color: #2c3e50; }}
    .header {{
      background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%);
      color: white; padding: 22px 32px;
      display: flex; align-items: center; justify-content: space-between;
    }}
    .header h1 {{ font-size: 20px; font-weight: 600; }}
    .header .sub {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}
    .kpi-row {{
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 14px; padding: 18px 24px 6px;
    }}
    .kpi {{
      background: white; border-radius: 8px; padding: 18px 22px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border-left: 4px solid var(--c);
    }}
    .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #7f8c8d; }}
    .kpi-value {{ font-size: 26px; font-weight: 700; color: var(--c); margin-top: 4px; }}
    .kpi-sub {{ font-size: 11px; color: #95a5a6; margin-top: 2px; }}
    .row2 {{
      display: grid; grid-template-columns: 3fr 2fr;
      gap: 14px; padding: 6px 24px;
    }}
    .row3 {{
      display: grid; grid-template-columns: 2fr 3fr;
      gap: 14px; padding: 6px 24px 20px;
    }}
    .card {{
      background: white; border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden;
    }}
    .footer {{
      text-align: center; padding: 10px; font-size: 11px; color: #aaa;
    }}
  </style>
</head>
<body>

<div class="header">
  <div>
    <h1>Boston Poverty Analysis</h1>
    <div class="sub">Suffolk County, MA — ACS 2014 Five-Year Estimates · Table B17021: Poverty Status by Living Arrangement</div>
  </div>
  <div style="font-size:12px;opacity:0.8;">Source: US Census Bureau</div>
</div>

<div class="kpi-row">
  <div class="kpi" style="--c:#2171b5">
    <div class="kpi-label">Overall Poverty Rate</div>
    <div class="kpi-value">{overall_rate:.1f}%</div>
    <div class="kpi-sub">Suffolk County aggregate</div>
  </div>
  <div class="kpi" style="--c:#e15759">
    <div class="kpi-label">People Below Poverty Line</div>
    <div class="kpi-value">{stats["total_poor"]:,}</div>
    <div class="kpi-sub">of {stats["total_pop"]:,} total</div>
  </div>
  <div class="kpi" style="--c:#59a14f">
    <div class="kpi-label">Census Tracts Analyzed</div>
    <div class="kpi-value">{stats["n_tracts"]}</div>
    <div class="kpi-sub">Avg {stats["mean"]:.1f}% · Median {stats["median"]:.1f}%</div>
  </div>
  <div class="kpi" style="--c:#f28e2b">
    <div class="kpi-label">Highest Tract Rate</div>
    <div class="kpi-value">{stats["max"]:.1f}%</div>
    <div class="kpi-sub">Std dev: ±{stats["std"]:.1f}%</div>
  </div>
</div>

<div class="row2">
  <div class="card">{map_html}</div>
  <div class="card">{hist_html}</div>
</div>

<div class="row3">
  <div class="card">{scatter_html}</div>
  <div class="card">{top15_html}</div>
</div>

<div class="footer">
  Python replication of R analysis by Susan Li &nbsp;·&nbsp;
  Data: US Census ACS 2014 &nbsp;·&nbsp;
  Visualization: Plotly
</div>

</body>
</html>"""
    return html


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Boston Poverty Dashboard — Python/Plotly")
    print("=" * 60)

    census_df = fetch_census_poverty_data()
    gdf = fetch_tiger_shapefile()
    merged = merge_spatial_data(census_df, gdf)

    print("\nComputing statistics...")
    stats = compute_stats(merged)
    overall_rate = 100 * stats["total_poor"] / stats["total_pop"]

    print("\nKey Statistics:")
    print(f"  Mean poverty rate:    {stats['mean']:.1f}%")
    print(f"  Median:               {stats['median']:.1f}%")
    print(f"  Highest tract:        {stats['max']:.1f}%")
    print(f"  Lowest tract:         {stats['min']:.1f}%")
    print(f"  Std deviation:        ±{stats['std']:.1f}%")
    print(f"  Total population:     {stats['total_pop']:,}")
    print(f"  People in poverty:    {stats['total_poor']:,}")
    print(f"  Overall poverty rate: {overall_rate:.1f}%")

    print("\nBuilding dashboard...")
    html = assemble_dashboard(merged, stats)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDashboard saved → {OUTPUT_FILE}")
    print("Open in your browser to explore the interactive charts.")


if __name__ == "__main__":
    main()
