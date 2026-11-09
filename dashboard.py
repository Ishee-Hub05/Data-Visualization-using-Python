import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

st.set_page_config(page_title="EDA Dashboard", layout="wide")
px.defaults.template = "plotly_dark"
MAX_POINTS = 8000  # to avoid browser freeze

@st.cache_data
def load_iris_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)
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
@st.cache_data
def compute_variances(df, cols):
    X = df[list(cols)].astype(float)
    rng = X.max() - X.min()
    X_scaled = (X - X.min()) / rng.replace(0, np.nan)
    return X_scaled.var(ddof=0).fillna(0.0)
@st.cache_data
def compute_feature_importance_and_metrics(df, feature_cols, target_col, is_classification):
    X = df[list(feature_cols)].copy()
    y = df[target_col].copy()
    X = X.fillna(X.median())
    
    metrics = {}
    if is_classification:
        if not pd.api.types.is_numeric_dtype(y):
            y_encoded, uniques = pd.factorize(y)
            y = pd.Series(y_encoded, index=y.index)
        else:
            y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
        if len(X) >= 10 and y.nunique() > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y if y.value_counts().min() >= 2 else None
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics["Accuracy"] = round(float(accuracy_score(y_test, y_pred)) * 100, 2)
            metrics["F1_Score"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 3)
            metrics["Type"] = "Classification"
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            metrics["Type"] = "Classification"
    else:
        y = y.fillna(y.median())
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics["R2_Score"] = round(float(r2_score(y_test, y_pred)), 3)
            metrics["RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
            metrics["Type"] = "Regression"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            metrics["Type"] = "Regression"
            
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=True)
    return imp_df, metrics

def main():
    st.title("Interactive EDA Dashboard")
    st.sidebar.header("Controls")
    data_src = st.sidebar.radio("Choose Data Source", ["Iris Dataset","Upload CSV"])
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
        selected_vals = st.sidebar.multiselect(f"Select {filt_col}",unique_vals,default=list(unique_vals))
        df_filt = apply_filt(df, filt_col, tuple(selected_vals))
    else:
        df_filt = df
    if df_filt.empty:
        st.warning("No data after filtering")
        return

    all_cols = df_filt.columns.tolist()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Target & ML Settings")
    target_col = st.sidebar.selectbox("Target Variable (Y)", all_cols, index=len(all_cols) - 1)
    var_thresh = st.sidebar.slider("Variance Threshold (normalized)",min_value=0.0,max_value=0.10,value=0.01,step=0.005,format="%.3f")
    exclude_low_var = st.sidebar.checkbox(f"Filter Low-Variance Features (variance <= {var_thresh:.3f})", value=False)

    st.markdown("Overview of data")
    with st.expander("Displaying first 10 rows"):
        st.dataframe(df_filt.head(10), use_container_width=True)
    with st.expander("Statistical Summary"):
        st.write(df_filt.describe())

    initial_missing_series = df_filt.isnull().sum()
    total_missing = int(initial_missing_series.sum())

    for col in df_filt.columns:
        if df_filt[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_filt[col]):
                df_filt[col] = df_filt[col].fillna(df_filt[col].median())
            else:
                mode_val = df_filt[col].mode()
                df_filt[col] = df_filt[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

    with st.expander("Data Cleaning & Missing Value Audit"):
        if total_missing == 0:
            st.success("No missing values found! The dataset is complete (100% integrity).")
        else:
            missing_cols = initial_missing_series[initial_missing_series > 0]
            missing_pct = (missing_cols / len(df_filt)) * 100
            missing_df = pd.DataFrame({
                "Column": missing_cols.index,
                "Missing Cells": missing_cols.values,
                "Percentage (%)": missing_pct.round(2).values
            }).sort_values(by="Missing Cells", ascending=False)
            st.dataframe(missing_df, use_container_width=True)
            st.success("Dataset cleaned: Numeric missing values imputed with median, categorical missing values imputed with mode.")
        csv_data = df_filt.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            help="Download the complete dataset after automated median/mode imputation."
        )
    variances = compute_variances(df_filt, tuple(num_cols))
    low_var_cols = [c for c in num_cols if variances[c] <= var_thresh]
    selectable_num_cols = num_cols.copy()
    if exclude_low_var and low_var_cols:
        filtered_cols = [c for c in num_cols if c not in low_var_cols]
        if filtered_cols:
            selectable_num_cols = filtered_cols

    with st.expander("Statistical Outlier Diagnostic (IQR)"):
        outlier_records = []
        for col in num_cols:
            q1 = df_filt[col].quantile(0.25)
            q3 = df_filt[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (df_filt[col] < lower_bound) | (df_filt[col] > upper_bound)
            count = int(outlier_mask.sum())
            if count > 0:
                pct = (count / len(df_filt)) * 100
                outlier_records.append({"Column": col,"Outlier Count": count,"Percentage (%)": round(pct, 2),"Normal Range (1.5 x IQR)": f"[{lower_bound:.2f}, {upper_bound:.2f}]"})
        if outlier_records:
            st.dataframe(pd.DataFrame(outlier_records), use_container_width=True)
            st.caption("Outliers are statistically detected using Tukey's Rule (1.5 x IQR). Use the Box Plot and Scatter Plot tabs to inspect them visually.")
        else:
            st.success("No statistical outliers detected across numerical columns (all points lie within 1.5 x IQR bounds).")

    with st.expander("📄 Export Full EDA Summary Report"):
        st.write("Generate an executive EDA audit report including dataset metadata, missing values, outlier diagnostics, target analysis, and feature variance.")
        outlier_summary_txt = "\n".join([f"  - {r['Column']}: {r['Outlier Count']} outliers ({r['Percentage (%)']}%) | Range: {r['Normal Range (1.5 x IQR)']}" for r in outlier_records]) if outlier_records else "  - None detected."
        missing_summary_txt = f"  - Total Missing Cells: {total_missing}\n" + ("\n".join([f"  - {idx}: {val} missing ({round(val/len(df_filt)*100, 2)}%)" for idx, val in initial_missing_series[initial_missing_series > 0].items()]) if total_missing > 0 else "  - Complete dataset (0 missing values).")
        
        report_content = f"""
EXPLORATORY DATA ANALYSIS & PRE-MODELING AUDIT REPORT
Generated via Interactive EDA Dashboard

1. Dataset Overview

- Total Records: {len(df_filt)}
- Total Features: {len(df_filt.columns)}
- Numeric Features ({len(num_cols)}): {', '.join(num_cols)}
- Categorical Features ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'None'}
- Target Variable Selected: {target_col}

2. Data Cleaning & Imputation Audit

{missing_summary_txt}
- Imputation Strategy: Numeric = Median, Categorical = Mode
- Post-Cleaning Missing Count: 0 cells (100% integrity)

3. Statistical Outlier Diagnostics (TUKEY'S 1.5 x IQR)

{outlier_summary_txt}

4. Target & Pre-Modeling Summary

- Problem Type: {'Classification' if (df_filt[target_col].nunique() <= 20 or not pd.api.types.is_numeric_dtype(df_filt[target_col])) else 'Regression'}
- Distinct Target Values: {df_filt[target_col].nunique()}
- Normalized Variance Threshold Applied: {var_thresh:.3f}
- Features Filtered by Low Variance: {', '.join(low_var_cols) if (exclude_low_var and low_var_cols) else 'None'}

End Of Report
"""
        st.download_button(label="📥 Download EDA Summary Report (.txt)",data=report_content,file_name="eda_summary_report.txt",mime="text/plain",help="Download a complete structured executive audit report of the dataset.")
    plot_df = df_filt
    if len(plot_df) > MAX_POINTS:
        plot_df = plot_df.sample(n=MAX_POINTS, random_state=42)
    x_col = st.selectbox("X-axis", selectable_num_cols)
    y_col = st.selectbox("Y-axis", selectable_num_cols, index=1 if len(selectable_num_cols) > 1 else 0)
    st.markdown("Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Relationships", "Correlation", "ML Pre-Modeling"])

    # Tab 1: Distributions
    with tab1:
        if st.checkbox("Show Histogram"):
            fig = px.histogram(plot_df, x=x_col, nbins=30)
            fig.update_traces(marker_line_color="#1E1E1E", marker_line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show Box Plot"):
            fig = px.box(plot_df, y=y_col)
            st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show Violin Plot"):
            fig = px.violin(plot_df, y=y_col, box=True, points="outliers")
            st.plotly_chart(fig, use_container_width=True)
        if cat_cols:
            if st.checkbox("Show Category Frequency Bar Chart"):
                cat_chart_col = st.selectbox("Select Categorical Feature", cat_cols, key="cat_count_select")
                counts_df = plot_df[cat_chart_col].value_counts().reset_index()
                counts_df.columns = [cat_chart_col, "Count"]
                fig_cat = px.bar(counts_df,x=cat_chart_col,y="Count",title=f"Frequency Count: {cat_chart_col}",text_auto=True)
                st.plotly_chart(fig_cat, use_container_width=True)

    # Tab 2: Relationship
    with tab2:
        if st.checkbox("Show Scatter Plot"):
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=target_col)
            st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show Line Chart"):
            fig = px.line(plot_df.sort_values(x_col), x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Correlation 
    with tab3:
        if st.checkbox("Show Correlation Heatmap"):
            corr = compute_corr(df_filt, tuple(num_cols))
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 4: ML Pre-Modeling
    with tab4:
        st.subheader("1. Target Analysis & Dataset Balance Check")
        st.write(f"Current Target Variable: **`{target_col}`** (change in sidebar)")
        is_class = df_filt[target_col].nunique() <= 20 or not pd.api.types.is_numeric_dtype(df_filt[target_col])
        if is_class:
            counts = df_filt[target_col].value_counts(dropna=False)
            pcts = df_filt[target_col].value_counts(normalize=True, dropna=False) * 100
            balance_df = pd.DataFrame({"Class": counts.index.astype(str), "Count": counts.values, "Percentage": pcts.values})
            majority_pct = pcts.iloc[0]
            if majority_pct <= 60:
                st.success(f"**Balanced Dataset:** The majority class is {majority_pct:.1f}% (<= 60%). Standard evaluation metrics (Accuracy) are reliable.")
            elif majority_pct <= 75:
                st.warning(f"**Moderately Imbalanced Dataset:** The majority class is {majority_pct:.1f}%. Consider stratified sampling during train-test split, evaluate with F1-Score/Balanced Accuracy instead of raw accuracy, and consider setting class_weight= balanced in tree/linear models")
            else:
                st.error(f"**Severely Imbalanced Dataset:** The majority class is {majority_pct:.1f}% (> 75%). Recommend using F1-Score, ROC-AUC, or resampling techniques (SMOTE).")

            fig_donut = px.pie(balance_df, names="Class", values="Count", hole=0.45, title=f"Class Distribution: {target_col}")
            fig_donut.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info(f"**Continuous Target Detected (Regression):** '{target_col}' has {df_filt[target_col].nunique()} distinct numerical values.")

        st.markdown("---")
        st.subheader("2. Feature Selection (Variance Threshold)")
        st.caption(f"Active Threshold: **variance <= {var_thresh:.3f}** on min-max scaled features (adjust in sidebar under Target & ML Settings)")
        candid_features = [c for c in num_cols if c != target_col]
        if candid_features:
            var_df = pd.DataFrame({
                "Feature": candid_features,
                "Normalized Variance": [round(float(variances[c]), 4) for c in candid_features],
            })
            var_df["Status"] = np.where(var_df["Normalized Variance"] > var_thresh, "Keep", "Drop (low variance)")
            low_var_feats = var_df.loc[var_df["Status"] != "Keep", "Feature"].tolist()
            if low_var_feats:
                st.warning(f"Found {len(low_var_feats)} low-variance feature(s) with variance <= {var_thresh:.3f}: {', '.join(low_var_feats)}")
            else:
                st.success(f"No low-variance features detected (all variances > {var_thresh:.3f}). Every numeric feature carries meaningful spread.")
            st.dataframe(var_df.sort_values(by="Normalized Variance"), use_container_width=True)

            fig_var = px.bar(var_df.sort_values(by="Normalized Variance", ascending=True), x="Normalized Variance", y="Feature", orientation="h", color="Status", color_discrete_map={"Keep": "#2ECC71", "Drop (low variance)": "#E74C3C"}, title="Normalized Feature Variance vs Threshold", text_auto=".3f")
            fig_var.add_vline(x=var_thresh, line_dash="dash", line_color="white", annotation_text="Threshold")
            st.plotly_chart(fig_var, use_container_width=True)
            st.caption("Variance is computed after min-max scaling (0-1) so that features measured in different units are compared fairly. A variance of 0 means the feature is constant.")
        else:
            st.info("No numeric predictor features are available for the variance check.")

        st.markdown("---")
        st.subheader("3. Feature Importance & Baseline Model Evaluation")
        if candid_features:
            if st.button("Calculate Feature Importance & Model Benchmark", type="primary"):
                with st.spinner("Training baseline Random Forest model and evaluating performance..."):
                    imp_df, metrics = compute_feature_importance_and_metrics(df_filt, tuple(candid_features), target_col, is_class)
                    if metrics.get("Type") == "Classification" and "Accuracy" in metrics:
                        st.markdown("#### Baseline Model Benchmark (Random Forest)")
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Validation Accuracy", f"{metrics['Accuracy']}%")
                        col_m2.metric("Weighted F1-Score", f"{metrics['F1_Score']}")
                        st.caption("Evaluated on a 25% holdout test set using 50 estimators.")
                    elif metrics.get("Type") == "Regression" and "R2_Score" in metrics:
                        st.markdown("#### Baseline Model Benchmark (Random Forest)")
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("R² Score (Variance Explained)", f"{metrics['R2_Score']}")
                        col_m2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']}")
                        st.caption("Evaluated on a 25% holdout test set using 50 estimators.")

                    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title=f"Random Forest Feature Importance for '{target_col}'", text_auto=".3f")
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.caption("Feature importance is derived from a baseline Random Forest model using Mean Decrease in Impurity (Gini).")
        else:
            st.warning("No numeric predictor features are available for Feature Importance calculation.")

    st.markdown("---")
    st.caption("Dashboard for every dataset • Cached • Safe • Stable")
if __name__ == "__main__":
    main()
