"""
Customer Segmentation Module using K-Means Clustering
------------------------------------------------------
Applies unsupervised K-Means clustering on online shoppers' behavioural features
to identify distinct shopper personas and purchase patterns.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Behavioural features relevant for segmentation
SEGMENTATION_FEATURES = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues'
]

def perform_kmeans_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]], KMeans, StandardScaler]:
    """
    Fits K-Means clustering on behavioural features and generates cluster profiles.

    Returns:
        df_clustered: Original DataFrame with an added 'Cluster' column.
        profile_df: DataFrame summarizing average feature values and conversion rates by cluster.
        cluster_info: Dictionary containing descriptive explanations for each segment.
        kmeans_model: Fitted KMeans model.
        scaler: Fitted StandardScaler.
    """
    X_seg = df[SEGMENTATION_FEATURES].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters

    # Convert Revenue to boolean if needed
    if df_clustered['Revenue'].dtype == object:
        rev_bool = df_clustered['Revenue'].astype(str).str.upper() == 'TRUE'
    else:
        rev_bool = df_clustered['Revenue'].astype(bool)
    df_clustered['_Revenue_Bool'] = rev_bool

    # Compute summary profile per cluster
    records = []
    cluster_info = {}

    for c in range(n_clusters):
        c_df = df_clustered[df_clustered['Cluster'] == c]
        count = len(c_df)
        pct = (count / len(df_clustered)) * 100.0
        conv_rate = (c_df['_Revenue_Bool'].mean()) * 100.0

        avg_pv = c_df['PageValues'].mean()
        avg_pr = c_df['ProductRelated'].mean()
        avg_dur = c_df['ProductRelated_Duration'].mean()
        avg_br = c_df['BounceRates'].mean()
        avg_er = c_df['ExitRates'].mean()
        avg_adm = c_df['Administrative'].mean()

        records.append({
            'Cluster': f"Segment {c}",
            'Sessions': count,
            'Audience Share (%)': round(pct, 1),
            'Conversion Rate (%)': round(conv_rate, 1),
            'Avg PageValues': round(avg_pv, 2),
            'Avg Product Pages': round(avg_pr, 1),
            'Avg Product Duration (s)': round(avg_dur, 1),
            'Avg Bounce Rate': round(avg_br, 4),
            'Avg Exit Rate': round(avg_er, 4),
            'Avg Admin Pages': round(avg_adm, 1),
        })

        # Infer descriptive characterization based on cluster metrics
        if avg_pv >= 15.0 or conv_rate >= 40.0:
            name = "High-Intent Buyers"
            desc = (
                f"Visitors with high average PageValues (${avg_pv:.2f}) and high conversion rate "
                f"({conv_rate:.1f}%). Highly qualified prospects with clear transactional intent."
            )
        elif avg_br >= 0.08 or avg_er >= 0.08:
            name = "Window Shoppers / Casual Browsers"
            desc = (
                f"Visitors characterized by high bounce rates ({avg_br:.3f}) and zero/low PageValues. "
                f"They leave after viewing few pages, resulting in low conversion ({conv_rate:.1f}%)."
            )
        elif avg_pr >= 35.0 or avg_dur >= 1200.0:
            name = "Deep Content Explorers"
            desc = (
                f"Visitors who view numerous product pages (avg {avg_pr:.0f}) and spend substantial time "
                f"({avg_dur:.0f}s). Moderate conversion ({conv_rate:.1f}%); they conduct extensive research."
            )
        else:
            name = "Brief / Standard Visitors"
            desc = (
                f"Visitors displaying standard browsing depth (avg {avg_pr:.0f} product pages) and "
                f"low administrative interaction. Moderate conversion baseline ({conv_rate:.1f}%)."
            )

        cluster_info[c] = {
            "name": name,
            "description": desc,
            "size": count,
            "share": pct,
            "conversion_rate": conv_rate
        }

    profile_df = pd.DataFrame(records)
    df_clustered.drop(columns=['_Revenue_Bool'], inplace=True)

    return df_clustered, profile_df, cluster_info, kmeans, scaler
