# Wikipedia ML Insights

> End-to-end data pipeline for clustering Wikipedia Machine Learning articles using Selenium, MongoDB, and Spark MLlib

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements a production-ready data pipeline that discovers meaningful patterns in Wikipedia articles through unsupervised machine learning.

**Key Achievement:** Identified 2 distinct article types with 0.84 silhouette score - "Regular Technical Articles" vs "Featured Encyclopedia Articles"

## Features

- Automated scraping with Selenium WebDriver
- MongoDB integration with automatic deduplication
- Apache Spark MLlib K-Means clustering
- 6 experimental validations
- 0% missing data across 250 articles

## Quick Start

### Prerequisites
```bash
Python 3.12+
Docker Desktop
Chrome Browser
```

### Installation
```bash
git clone https://github.com/Giotros/wikipedia-ml-insights.git
cd wikipedia-ml-insights

pip install -r requirements.txt
docker-compose up -d
```

### Run Pipeline
```bash
# Full pipeline (~40 minutes)
python wikipedia_scraper.py
python spark_clustering.py

# Or use existing data (5 minutes)
python spark_clustering.py
```

## Results

| Metric | Value |
|--------|-------|
| Dataset Size | 250 articles |
| Optimal k | 2 |
| Silhouette Score | 0.8410 (excellent) |
| Execution Time | ~40 minutes |

### Discovered Clusters

**Cluster 0 (n=241): Regular Technical Articles**
- Average length: 13,520 characters
- Average references: 18.5
- Examples: "Deep learning", "Neural network"

**Cluster 1 (n=9): Featured Encyclopedia Articles**
- Average length: 89,234 characters (6.6x larger)
- Average references: 256.8 (13.9x more)
- Examples: "Artificial intelligence", "Machine learning"

## Technology Stack

- **Scraping:** Selenium, Chrome WebDriver
- **Storage:** MongoDB, pymongo
- **Processing:** Apache Spark 3.5.0, MLlib
- **Language:** Python 3.12

## Project Structure
```
wikipedia-ml-insights/
├── wikipedia_scraper.py       # Selenium scraper
├── spark_clustering.py        # K-Means clustering
├── experiments.py             # 6 experiments
├── wikipedia_features.jsonl   # Dataset
└── docker-compose.yml         # MongoDB setup
```

## Key Findings

1. **StandardScaler Critical** - 159% silhouette improvement
2. **Wikipedia Stable** - Strict waits unnecessary (31% speedup)
3. **MongoDB Upsert Essential** - Guarantees idempotency
4. **Spark Optimal at Hardware Level** - 8 partitions = 8 cores

## Documentation

See [REPORT.md](REPORT.md) for detailed technical documentation including methodology, experiments, and analysis.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Citation
```
Wikipedia ML Insights: End-to-end clustering pipeline
Year: 2026
URL: https://github.com/Giotros/wikipedia-ml-insights
```
