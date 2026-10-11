import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

st.set_page_config(page_title="EDA Dashboard", layout="wide")
px.defaults.template = "plotly_dark"
MAX_POINTS = 8000  # to avoid browser freeze

@st.cache_data
def load_iris_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(
        data.target, data.target_names
    )
    return df

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def apply_filt(df, col, values):
    return df[df[col].isin(values)]

@st.cache_data
def compute_corr(df, cols):
    return df[list(cols)].corr()

def main():
    st.title("Interactive EDA Dashboard")
    st.sidebar.header("Controls")
    data_src = st.sidebar.radio( "Choose Data Source", ["Iris Dataset","Upload CSV"])
    if data_src == "Iris Dataset":
        df = load_iris_data()
    else:
        upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if upload is None:
            st.warning("Upload a CSV to continue")
            return
        df = load_csv(upload)
    if df.empty:
        st.error("Your Dataset is empty")
        return
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(exclude=np.number).columns if df[c].nunique() <= 100]
    if not num_cols:
        st.error("No numeric columns available")
        return
    if cat_cols:
        filt_col = st.sidebar.selectbox("Filter column", cat_cols)
        unique_vals = df[filt_col].unique()
        selected_vals = st.sidebar.multiselect(
            f"Select {filt_col}",
            unique_vals,
            default=list(unique_vals)
        )
        df_filt = apply_filt(df, filt_col, tuple(selected_vals))
    else:
        df_filt = df
    if df_filt.empty:
        st.warning("No data after filtering")
        return

    st.markdown("Overview of data")
    with st.expander("Displaying first 10 rows"):
        st.dataframe(df_filt.head(10), use_container_width=True)
    with st.expander("Statistical Summary"):
        st.write(df_filt.describe())
    plot_df = df_filt
    if len(plot_df) > MAX_POINTS:
        plot_df = plot_df.sample(n=MAX_POINTS, random_state=42)
    x_col = st.selectbox("X-axis", num_cols)
    y_col = st.selectbox("Y-axis", num_cols, index=1 if len(num_cols) > 1 else 0)

    st.markdown("Visualizations")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Correlation"])
    #Tab 1: Distributions
    with tab1:
        if st.checkbox("Show Histogram"):
            fig = px.histogram(plot_df, x=x_col, nbins=30)
            fig.update_traces(marker_line_color="#1E1E1E", marker_line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show Box Plot"):
            fig = px.box(plot_df, y=y_col)
            st.plotly_chart(fig, use_container_width=True)
    #Tab 2: Relationship
    with tab2:
        if st.checkbox("Show Scatter Plot"):
            fig = px.scatter(plot_df, x=x_col, y=y_col, render_mode= "webgl")
            st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show Line Chart"):
            fig = px.line(plot_df.sort_values(x_col), x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show Bar Chart"):
            fig = px.bar(plot_df, x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)
    #Tab 3: Correlation
    with tab3:
        if st.checkbox("Show Correlation Heatmap"):
            corr = compute_corr(df_filt, tuple(num_cols))
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("Dashboard for every dataset • Cached • Safe • Stable")

if __name__ == "__main__":
    main()
