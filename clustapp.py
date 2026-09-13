import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="SF Building Permits Clustering",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("San Francisco Building Permits – Clustering Explorer")
st.markdown("Interactive analysis of permit clusters based on project size, cost and type.")

# LOAD & PREPARE DATA 
@st.cache_data
def load_and_cluster(file_path="Building_Permits.csv"):
    df = pd.read_csv(file_path, low_memory=False)

    # Basic cleaning
    df = df[df["Revised Cost"] > 1000].copy()

    numeric_cols = [
        "Number of Existing Stories", "Number of Proposed Stories",
        "Existing Units", "Proposed Units", "Plansets",
        "Existing Construction Type", "Proposed Construction Type",
        "Estimated Cost", "Revised Cost"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = [
        "Permit Type Definition",
        "Existing Use", "Proposed Use",
        "Existing Construction Type Description",
        "Proposed Construction Type Description"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    features = [
        "Number of Existing Stories", "Number of Proposed Stories",
        "Existing Units", "Proposed Units", "Plansets",
        "Estimated Cost", "Revised Cost",
        "Existing Construction Type", "Proposed Construction Type",
        "Permit Type Definition", "Existing Use", "Proposed Use",
        "Existing Construction Type Description",
        "Proposed Construction Type Description"
    ]

    df_cluster = df[features + ["Neighborhoods - Analysis Boundaries"]].dropna().copy()

    X = df_cluster[features]
    neighborhoods = df_cluster["Neighborhoods - Analysis Boundaries"]

    numeric_features = [
        "Number of Existing Stories", "Number of Proposed Stories",
        "Existing Units", "Proposed Units", "Plansets",
        "Estimated Cost", "Revised Cost",
        "Existing Construction Type", "Proposed Construction Type"
    ]
    categorical_features = [
        "Permit Type Definition", "Existing Use", "Proposed Use",
        "Existing Construction Type Description",
        "Proposed Construction Type Description"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=3, random_state=42, n_init=10))
    ])

    pipeline.fit(X)
    labels = pipeline.predict(X)

    df_cluster = df_cluster.copy()
    df_cluster["Cluster"] = labels

    # PCA for visualization
    X_transformed = pipeline.named_steps["preprocessor"].transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_transformed)
    df_cluster["PCA1"] = X_pca[:, 0]
    df_cluster["PCA2"] = X_pca[:, 1]

    return df_cluster, pipeline, features, numeric_features, categorical_features

# Load data
with st.spinner("Loading data and running clustering..."):
    try:
        df, pipeline, features, numeric_features, categorical_features = load_and_cluster()
    except FileNotFoundError:
        st.error("❌ Could not find `Building_Permits.csv`. Please place it in the same folder as this app.")
        st.stop()

# SIDEBAR
st.sidebar.header("Filters & Controls")
selected_neighborhood = st.sidebar.selectbox(
    "Filter by Neighborhood (optional)",
    options=["All"] + sorted(df["Neighborhoods - Analysis Boundaries"].dropna().unique().tolist())
)

if selected_neighborhood != "All":
    df_view = df[df["Neighborhoods - Analysis Boundaries"] == selected_neighborhood]
else:
    df_view = df

st.sidebar.markdown("---")
st.sidebar.markdown("**Cluster Legend**")
st.sidebar.markdown("🟢 **Cluster 0** – Everyday / Residential")
st.sidebar.markdown("🔵 **Cluster 1** – Mid-to-High Rise")
st.sidebar.markdown("🔴 **Cluster 2** – Mega High-Rise")

# 1. CLUSTER OVERVIEW

st.header("1. Cluster Overview")

cluster_summary = df.groupby("Cluster").agg({
    "Revised Cost": "mean",
    "Estimated Cost": "mean",
    "Number of Proposed Stories": "mean",
    "Proposed Units": "mean",
    "Number of Existing Stories": "mean",
    "Existing Units": "mean",
    "Plansets": "mean"
}).round(1)

cluster_counts = df["Cluster"].value_counts().sort_index()
cluster_summary["Count"] = cluster_counts
cluster_summary["Percentage"] = (cluster_counts / len(df) * 100).round(1)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Cluster 0 – Everyday", f"{cluster_counts[0]:,}", f"{cluster_summary.loc[0, 'Percentage']}%")
    st.write(f"Avg Cost: **${cluster_summary.loc[0, 'Revised Cost']:,.0f}**")
    st.write(f"Avg Stories: **{cluster_summary.loc[0, 'Number of Proposed Stories']}**")
    st.write(f"Avg Units: **{cluster_summary.loc[0, 'Proposed Units']}**")

with col2:
    st.metric("Cluster 1 – Mid/High-Rise", f"{cluster_counts[1]:,}", f"{cluster_summary.loc[1, 'Percentage']}%")
    st.write(f"Avg Cost: **${cluster_summary.loc[1, 'Revised Cost']:,.0f}**")
    st.write(f"Avg Stories: **{cluster_summary.loc[1, 'Number of Proposed Stories']}**")
    st.write(f"Avg Units: **{cluster_summary.loc[1, 'Proposed Units']}**")

with col3:
    st.metric("Cluster 2 – Mega High-Rise", f"{cluster_counts[2]:,}", f"{cluster_summary.loc[2, 'Percentage']}%")
    st.write(f"Avg Cost: **${cluster_summary.loc[2, 'Revised Cost']:,.0f}**")
    st.write(f"Avg Stories: **{cluster_summary.loc[2, 'Number of Proposed Stories']}**")
    st.write(f"Avg Units: **{cluster_summary.loc[2, 'Proposed Units']}**")

st.dataframe(cluster_summary, use_container_width=True)

# 2. NEIGHBORHOOD BREAKDOWN
st.header("2. Neighborhood Breakdown by Cluster")

top_n = st.slider("Show top N neighborhoods per cluster", 5, 15, 8)

for cluster_id in [0, 1, 2]:
    st.subheader(f"Cluster {cluster_id}")
    subset = df[df["Cluster"] == cluster_id]
    top_neigh = (
        subset["Neighborhoods - Analysis Boundaries"]
        .value_counts(normalize=True)
        .head(top_n)
        .mul(100)
        .reset_index()
    )
    top_neigh.columns = ["Neighborhood", "Percentage"]

    fig = px.bar(
        top_neigh,
        x="Percentage",
        y="Neighborhood",
        orientation="h",
        color="Percentage",
        color_continuous_scale="Teal",
        text=top_neigh["Percentage"].round(1).astype(str) + "%"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# 3. INTERACTIVE NEIGHBORHOOD FILTER
st.header("3. Neighborhood Deep Dive")

if selected_neighborhood != "All":
    st.info(f"Showing results only for **{selected_neighborhood}**")
    cluster_dist = df_view["Cluster"].value_counts(normalize=True).sort_index().mul(100)
    fig = px.pie(
        names=[f"Cluster {i}" for i in cluster_dist.index],
        values=cluster_dist.values,
        color_discrete_sequence=["#0D9488", "#3B82F6", "#EF4444"],
        title=f"Cluster distribution in {selected_neighborhood}"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Select a neighborhood in the sidebar to see its cluster distribution.")

# 5. PREDICT CLUSTER FOR A NEW PERMIT
st.header("5. Which cluster would this permit belong to?")

st.markdown("Enter the characteristics of a new permit and the model will assign it to a cluster.")

with st.form("predict_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        stories_existing = st.number_input("Existing Stories", min_value=0.0, value=2.0, step=1.0)
        stories_proposed = st.number_input("Proposed Stories", min_value=0.0, value=3.0, step=1.0)
        units_existing = st.number_input("Existing Units", min_value=0.0, value=2.0, step=1.0)
        units_proposed = st.number_input("Proposed Units", min_value=0.0, value=4.0, step=1.0)
        plansets = st.number_input("Plansets", min_value=0.0, value=2.0, step=1.0)

    with col_b:
        est_cost = st.number_input("Estimated Cost ($)", min_value=1000.0, value=50000.0, step=1000.0)
        rev_cost = st.number_input("Revised Cost ($)", min_value=1000.0, value=55000.0, step=1000.0)
        exist_const = st.number_input("Existing Construction Type (1-5)", min_value=1.0, max_value=5.0, value=5.0, step=1.0)
        prop_const = st.number_input("Proposed Construction Type (1-5)", min_value=1.0, max_value=5.0, value=5.0, step=1.0)

    permit_type = st.selectbox("Permit Type Definition", options=df["Permit Type Definition"].value_counts().head(10).index.tolist())
    existing_use = st.selectbox("Existing Use", options=df["Existing Use"].value_counts().head(10).index.tolist())
    proposed_use = st.selectbox("Proposed Use", options=df["Proposed Use"].value_counts().head(10).index.tolist())

    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    new_data = pd.DataFrame([{
        "Number of Existing Stories": stories_existing,
        "Number of Proposed Stories": stories_proposed,
        "Existing Units": units_existing,
        "Proposed Units": units_proposed,
        "Plansets": plansets,
        "Estimated Cost": est_cost,
        "Revised Cost": rev_cost,
        "Existing Construction Type": exist_const,
        "Proposed Construction Type": prop_const,
        "Permit Type Definition": permit_type,
        "Existing Use": existing_use,
        "Proposed Use": proposed_use,
        "Existing Construction Type Description": "Unknown",
        "Proposed Construction Type Description": "Unknown"
    }])

    pred = pipeline.predict(new_data)[0]
    cluster_names = {
        0: "🟢 Cluster 0 – Everyday / Residential",
        1: "🔵 Cluster 1 – Mid-to-High Rise",
        2: "🔴 Cluster 2 – Mega High-Rise"
    }
    st.success(f"**Predicted Cluster:** {cluster_names[pred]}")

# 6. INTERACTIVE PCA PLOT
st.header("6. PCA Interactive Scatter Plot")

sample_size = st.slider("Number of points to display ", 2000, 80000, 8000, step=5000)
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

fig_pca = px.scatter(
    df_sample,
    x="PCA1",
    y="PCA2",
    color=df_sample["Cluster"].astype(str),
    color_discrete_map={"0": "#0D9488", "1": "#3B82F6", "2": "#EF4444"},
    opacity=0.6,
    title="Clusters visualized with PCA",
    labels={"color": "Cluster"},
    hover_data=["Revised Cost", "Number of Proposed Stories", "Proposed Units"]
)
fig_pca.update_layout(height=550)
st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("---")
st.caption("K-Means Clustering on San Francisco Building Permits")