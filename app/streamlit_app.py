"""Child survival and the 2030 target: interactive app.

Run with:  streamlit run app/streamlit_app.py
"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQUA, BLUE, GREY, ORANGE, bar, lines, load, predict, show, style

st.set_page_config(page_title="Child survival and the 2030 target", layout="wide")
model, s = load("model.json"), load("summary.json")
world = s["world"]
last = s["last_year"]
sdg = {row["iso3"]: row for row in s["sdg"]}
profiles = {p["iso3"]: p for p in s["profiles"]}
years = s["map"]["years"]

# (feature, label, slider step, show on a natural scale, convert back)
SLIDERS = [
    ("basic_sanitation", "Basic sanitation (% of people)", 1.0, False),
    ("basic_water", "Basic drinking water (% of people)", 1.0, False),
    ("dpt_immunisation", "DPT immunisation (% of infants)", 1.0, False),
    ("electricity", "Access to electricity (% of people)", 1.0, False),
    ("fertility", "Births per woman", 0.1, False),
    ("undernourishment", "Undernourishment (% of people)", 0.5, False),
    ("log_gdp", "GDP per capita (PPP dollars)", 100.0, True),
    ("log_health_spending", "Health spending per person (US dollars)", 5.0, True),
]


def estimate(features):
    return math.exp(predict(model, [features[name] for name in model["features"]]))


st.title("Who will miss the 2030 target for child survival?")
st.caption("Every country has committed to an under-five mortality rate of 25 or fewer deaths per 1,000 live births by 2030. "
           "Data: World Bank World Development Indicators.")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Under-five deaths in {world[-1]['year']}", f"{world[-1]['deaths'] / 1e6:.1f}m", f"{world[0]['deaths'] / 1e6:.1f}m in {world[0]['year']}", delta_color="off")
c2.metric("World mortality rate", f"{world[-1]['u5mr']:.0f} per 1,000", f"{world[0]['u5mr']:.0f} in {world[0]['year']}", delta_color="off")
c3.metric("Countries off track for 2030", s["sdg_counts"]["Off track"], f"of {s['countries']}", delta_color="off")
c4.metric("Share of deaths in ten countries", f"{s['top10_share']:.0%}")

tab_map, tab_track, tab_whatif, tab_world, tab_model = st.tabs(["World map", "2030 tracker", "What-if explorer", "Global picture", "What goes with survival"])

with tab_map:
    year = st.slider("Year", years[0], years[-1], years[-1])
    column = years.index(year)
    figure = go.Figure(go.Choropleth(locations=s["map"]["iso3"], z=[row[column] for row in s["map"]["values"]], text=s["map"]["names"],
                                     colorscale="Blues", zmin=0, zmax=150, colorbar=dict(title="per 1,000"),
                                     hovertemplate="%{text}<br>%{z:.0f} deaths per 1,000 live births<extra></extra>"))
    figure.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False)
    show(style(figure, 520))

with tab_track:
    names = sorted((row["country"], iso) for iso, row in sdg.items())
    choice = st.selectbox("Country", [n for n, _ in names], index=[n for n, _ in names].index("Nigeria"))
    row = sdg[dict(names)[choice]]
    a, b, c = st.columns(3)
    label = row["status"] + (" · too close to call" if row["borderline"] else "")
    a.metric(f"Rate in {last}", f"{row['u5mr']:.0f} per 1,000", label, delta_color="off")
    b.metric("Projected 2030", f"{row['projected_2030']:.0f} per 1,000")
    c.metric("Yearly reduction", f"{row['arr']:.1%}", "target met" if row["status"] == "Target met" else f"{row['required_arr']:.1%} needed", delta_color="off")
    history = s["map"]["values"][s["map"]["iso3"].index(row["iso3"])]
    figure = lines([("Estimated rate", years, history, BLUE, "solid"), ("Projection on current trend", [last, 2030], [row["u5mr"], row["projected_2030"]], ORANGE, "dash")],
                   "", "Deaths per 1,000 live births")
    figure.add_hline(y=25, line_dash="dot", line_color=GREY, annotation_text="Target: 25")
    show(figure)
    if row["status"] != "Target met":
        st.write(f"How much does the choice of trend matter? On a 5-year trend the 2030 projection is {row['projected_5y']:.0f}, on the 10-year trend used here "
                 f"{row['projected_2030']:.0f}, and on a 15-year trend {row['projected_15y']:.0f}."
                 + (" This country is projected within five points of the target, so it could plausibly land on either side." if row["borderline"] else ""))
    status = st.selectbox("Show countries that are", ["Off track", "On track", "Target met", "Too close to call"])
    chosen = [r for r in s["sdg"] if (r["borderline"] if status == "Too close to call" else r["status"] == status)]
    table = pd.DataFrame(chosen).sort_values("projected_2030", ascending=False)
    table = table[["country", "region", "u5mr", "arr", "projected_2030", "projected_5y", "projected_15y", "borderline", "required_arr"]]
    table = table.rename(columns={"projected_5y": "on 5-year trend", "projected_15y": "on 15-year trend", "borderline": "too close to call"})
    st.dataframe(table.style.format({"u5mr": "{:.0f}", "arr": "{:.1%}", "projected_2030": "{:.0f}", "on 5-year trend": "{:.0f}", "on 15-year trend": "{:.0f}",
                                     "required_arr": "{:.1%}"}), hide_index=True, width="stretch")
    st.markdown("**Does the trend window change the picture?** Number of countries in each status, by the window used to estimate the trend.")
    st.dataframe(pd.DataFrame(s["sensitivity"]), hide_index=True, width="stretch")
    st.caption("Projections are single values without an uncertainty range. The UN's own uncertainty ranges around each mortality estimate are not carried into the projection.")

with tab_whatif:
    st.caption("Pick a country and change its indicators. The model estimates the mortality rate typical of countries with that profile. "
               "This shows associations across countries. It is not a forecast of what a policy would achieve.")
    options = sorted((p["country"], iso) for iso, p in profiles.items())
    picked = st.selectbox("Country", [n for n, _ in options], index=[n for n, _ in options].index("Nigeria"), key="whatif_country")
    profile = profiles[dict(options)[picked]]
    features = dict(profile["features"])
    baseline = estimate(features)
    columns = st.columns(2)
    for i, (key, label, step, is_log) in enumerate(SLIDERS):
        value = profile["features"][key]
        if value is None:
            columns[i % 2].caption(f"{label}: no data for {picked}")
            continue
        low, high = s["ranges"][key]
        shown = math.exp(value) if is_log else value
        low_shown, high_shown = (math.exp(low), math.exp(high)) if is_log else (low, high)
        low_shown, high_shown = min(low_shown, shown), max(high_shown, shown)
        chosen = columns[i % 2].slider(label, float(math.floor(low_shown)), float(math.ceil(high_shown)), float(shown), step, key=f"{picked}_{key}")
        if abs(chosen - shown) > step / 2:
            features[key] = math.log(max(chosen, step)) if is_log else chosen
    scenario = estimate(features)
    st.divider()
    left, right = st.columns([1, 2])
    change = scenario / baseline - 1
    left.metric("Model estimate for this profile", f"{scenario:.0f} per 1,000", "current profile" if abs(change) < 0.005 else f"{change:+.0%} against current profile", delta_color="off")
    right.write(f"**{picked}:** the UN estimate for {profile['year']} is {profile['u5mr']:.0f} per 1,000. "
                f"Countries with {picked}'s current indicators typically have about {baseline:.0f}.")
    right.write("A country with the adjusted profile would typically be " + ("at or below" if scenario <= 25 else "still above") + " the 2030 target of 25.")

with tab_world:
    left, right = st.columns(2)
    with left:
        st.markdown("**World under-five mortality rate**")
        figure = lines([("World rate", [r["year"] for r in world], [r["u5mr"] for r in world], BLUE, "solid")], "", "Deaths per 1,000 live births")
        figure.add_hline(y=25, line_dash="dot", line_color=GREY, annotation_text="Target: 25")
        show(figure)
        st.markdown("**Ten countries with the most deaths** (thousands)")
        top = s["top_countries"][::-1]
        show(bar([r["country"] for r in top], [r["u5_deaths"] / 1000 for r in top], f"Under-five deaths, {last} (thousands)", "", horizontal=True))
    with right:
        st.markdown("**Where children die** (share of deaths by region)")
        regions = s["regions"][::-1]
        show(bar([r["region"] for r in regions], [r["share_of_deaths"] * 100 for r in regions], "Share of under-five deaths (%)", "", horizontal=True))
        st.markdown("**Countries by 2030 status**")
        order = ["Target met", "On track", "Off track"]
        show(bar(order, [s["sdg_counts"].get(k, 0) for k in order], "", "Countries"))

with tab_model:
    st.caption("A model trained on 13 development indicators and tested on countries it had not seen.")
    st.dataframe(pd.DataFrame(s["metrics"]).round(3), hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**How much each indicator moves the prediction**")
        importance = s["importance"][::-1]
        show(bar([r["feature"] for r in importance], [r["value"] for r in importance], "Mean absolute SHAP value", "", horizontal=True, height=440))
        st.markdown("**Lower mortality than predicted**")
        better = s["better"][::-1]
        show(bar([r["country"] for r in better], [(r["ratio"] - 1) * 100 for r in better], "Actual relative to prediction (%)", "", horizontal=True, color=AQUA))
    with right:
        st.markdown("**Predicted mortality as one indicator changes**")
        titles = {"basic_sanitation": "Basic sanitation (% of people)", "fertility": "Births per woman",
                  "dpt_immunisation": "DPT immunisation (% of infants)", "log_gdp": "GDP per capita (PPP dollars)"}
        key = st.selectbox("Indicator", list(titles), format_func=titles.get)
        figure = lines([("Predicted mortality", s["curves"][key]["x"], s["curves"][key]["y"], BLUE, "solid")], titles[key], "Predicted deaths per 1,000", height=355)
        if key == "log_gdp":
            figure.update_xaxes(type="log")
        show(figure)
        st.markdown("**Higher mortality than predicted**")
        worse = s["worse"][::-1]
        show(bar([r["country"] for r in worse], [(r["ratio"] - 1) * 100 for r in worse], "Actual relative to prediction (%)", "", horizontal=True, color=ORANGE))
