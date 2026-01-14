# Technical Report: Wikipedia ML Clustering

**Completion Date:** January 2026

## Overview

End-to-end pipeline for clustering Wikipedia Machine Learning articles using Selenium, MongoDB, and Spark MLlib.

## Results

- **Dataset:** 250 articles, 0% missing data
- **Optimal k:** 2 clusters
- **Silhouette Score:** 0.8410 (excellent)
- **Execution Time:** ~40 minutes

## Clusters

**Cluster 0 (96%):** Regular technical articles
- Length: 13,520 chars avg
- References: 18.5 avg

**Cluster 1 (4%):** Featured articles  
- Length: 89,234 chars avg (6.6x larger)
- References: 256.8 avg (13.9x more)

## Key Findings

1. StandardScaler critical (+159% quality)
2. Wikipedia stable (strict waits unnecessary)
3. MongoDB upsert essential (idempotency)
4. Spark optimal at core count

## Technologies

Python 3.12, Selenium, MongoDB, Apache Spark 3.5.0, MLlib
