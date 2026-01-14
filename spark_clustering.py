from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import mean
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
import json


class WikipediaClusterer:
    """K-Means clustering για Wikipedia άρθρα"""
    
    def __init__(self, app_name: str = "WikipediaClustering", use_scaling: bool = True):
        """
        Args:
            app_name: Όνομα Spark application
            use_scaling: Αν True, εφαρμογή StandardScaler (πειραματισμός 6)
        """
        self.use_scaling = use_scaling
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        self.df = None
        self.feature_cols = [
            'page_length_chars',
            'num_sections',
            'num_references',
            'num_categories',
            'num_images',
            'num_infobox_rows'
        ]
        
    def load_data(self, jsonl_file: str = "wikipedia_features.jsonl"):
        """
        Φόρτωση JSONL δεδομένων στο Spark
        
        Args:
            jsonl_file: Path to JSONL file
        """
        print(f"Φόρτωση δεδομένων από {jsonl_file}...")
        self.df = self.spark.read.json(jsonl_file)
        
        print(f"Φορτώθηκαν {self.df.count()} εγγραφές")
        print("\nSchema:")
        self.df.printSchema()
        
        return self.df
    
    def clean_data(self):
        """
        Καθαρισμός δεδομένων:
        - Μετατροπή τύπων σε numeric
        - Χειρισμός nulls
        - Φιλτράρισμα ακραίων τιμών
        """
        print("\nΚαθαρισμός δεδομένων...")
        
        # Μετατροπή σε numeric
        for col_name in self.feature_cols:
            self.df = self.df.withColumn(
                col_name,
                col(col_name).cast(DoubleType())
            )
        
        # Χειρισμός nulls - αντικατάσταση με 0
        for col_name in self.feature_cols:
            self.df = self.df.withColumn(
                col_name,
                when(col(col_name).isNull(), 0.0).otherwise(col(col_name))
            )
        
        # Φιλτράρισμα ακραίων τιμών (άρθρα με μηδενικό content)
        self.df = self.df.filter(col('page_length_chars') > 100)
        
        print(f"Εγγραφές μετά τον καθαρισμό: {self.df.count()}")
        
        # Βασικά στατιστικά
        print("\nΒασικά στατιστικά χαρακτηριστικών:")
        self.df.select(self.feature_cols).describe().show()
        
        return self.df
    
    def create_features(self):
        """
        Δημιουργία feature vector με VectorAssembler
        και προαιρετικό scaling
        """
        print("\nΔημιουργία feature vectors...")
        
        # VectorAssembler
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features_raw"
        )
        
        self.df = assembler.transform(self.df)
        
        if self.use_scaling:
            print("Εφαρμογή StandardScaler...")
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=True
            )
            scaler_model = scaler.fit(self.df)
            self.df = scaler_model.transform(self.df)
        else:
            print("Χωρίς scaling - χρήση raw features")
            self.df = self.df.withColumnRenamed("features_raw", "features")
        
        return self.df
    
    def find_optimal_k(self, k_range: range = range(2, 9)):
        """
        Εύρεση βέλτιστου k με silhouette score
        
        Args:
            k_range: Εύρος τιμών k προς δοκιμή
            
        Returns:
            Dictionary με αποτελέσματα για κάθε k
        """
        print(f"\nΕύρεση βέλτιστου k (εύρος: {k_range.start}-{k_range.stop-1})...")
        
        results = []
        evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol='features',
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )
        
        for k in k_range:
            print(f"\nΔοκιμή k={k}...")
            
            # K-Means
            kmeans = KMeans(
                k=k,
                seed=42,
                featuresCol='features',
                predictionCol='prediction'
            )
            
            model = kmeans.fit(self.df)
            predictions = model.transform(self.df)
            
            # Silhouette score
            silhouette = evaluator.evaluate(predictions)
            
            # Within Set Sum of Squared Errors
            wssse = model.summary.trainingCost
            
            # Cluster sizes
            cluster_sizes = predictions.groupBy('prediction').count().collect()
            cluster_distribution = {row['prediction']: row['count'] for row in cluster_sizes}
            
            result = {
                'k': k,
                'silhouette': silhouette,
                'wssse': wssse,
                'cluster_sizes': cluster_distribution
            }
            results.append(result)
            
            print(f"  Silhouette: {silhouette:.4f}")
            print(f"  WSSSE: {wssse:.2f}")
            print(f"  Cluster sizes: {cluster_distribution}")
        
        # Αποθήκευση αποτελεσμάτων
        results_df = pd.DataFrame(results)
        results_df.to_csv('kmeans_evaluation.csv', index=False)
        print(f"\n✓ Αποτελέσματα αποθηκεύτηκαν στο kmeans_evaluation.csv")
        
        # Βέλτιστο k (μέγιστο silhouette)
        best_k = max(results, key=lambda x: x['silhouette'])['k']
        print(f"\n✓ Βέλτιστο k: {best_k}")
        
        return results, best_k
    
    def final_clustering(self, k: int):
        """
        Τελικό clustering με επιλεγμένο k
        
        Args:
            k: Αριθμός clusters
            
        Returns:
            Predictions DataFrame
        """
        print(f"\n{'='*60}")
        print(f"ΤΕΛΙΚΟ CLUSTERING ΜΕ k={k}")
        print(f"{'='*60}")
        
        kmeans = KMeans(
            k=k,
            seed=42,
            featuresCol='features',
            predictionCol='cluster'
        )
        
        model = kmeans.fit(self.df)
        predictions = model.transform(self.df)
        
        # Cluster centers
        centers = model.clusterCenters()
        print("\nCluster Centers:")
        for i, center in enumerate(centers):
            print(f"  Cluster {i}: {center}")
        
        return predictions, model
    
    def profile_clusters(self, predictions):
        """
        Profiling clusters - υπολογισμός μέσων τιμών και παραδείγματα
        
        Args:
            predictions: DataFrame με cluster assignments
        """
        print(f"\n{'='*60}")
        print("CLUSTER PROFILING")
        print(f"{'='*60}")
        
        # Μέσες τιμές ανά cluster
        cluster_profiles = predictions.groupBy('cluster').agg(
            *[mean(col(f)).alias(f) for f in self.feature_cols]
        ).toPandas()
        
        # Υπολογισμός μέσων τιμών
        for cluster_id in sorted(cluster_profiles['cluster'].unique()):
            cluster_data = predictions.filter(col('cluster') == cluster_id)
            cluster_size = cluster_data.count()
            
            print(f"\n{'─'*60}")
            print(f"CLUSTER {cluster_id} (n={cluster_size})")
            print(f"{'─'*60}")
            
            # Μέσες τιμές
            means = cluster_data.select(self.feature_cols).describe().filter(
                col('summary') == 'mean'
            ).collect()[0]
            
            print("Μέσες τιμές χαρακτηριστικών:")
            for i, col_name in enumerate(self.feature_cols, 1):
                print(f"  {col_name}: {float(means[i]):.2f}")
            
            # Ενδεικτικά άρθρα
            print("\nΕνδεικτικά άρθρα:")
            samples = cluster_data.select('title', 'url', 'page_length_chars', 'num_sections') \
                .limit(3).collect()
            
            for j, sample in enumerate(samples, 1):
                print(f"  {j}. {sample['title']}")
                print(f"     Length: {sample['page_length_chars']}, "
                      f"Sections: {sample['num_sections']}")
            
            # Ερμηνευτική ετικέτα (απλοποιημένη λογική)
            avg_length = float(means[self.feature_cols.index('page_length_chars') + 1])
            avg_refs = float(means[self.feature_cols.index('num_references') + 1])
            avg_infobox = float(means[self.feature_cols.index('num_infobox_rows') + 1])
            
            label = self._interpret_cluster(avg_length, avg_refs, avg_infobox)
            print(f"\nΕρμηνεία: {label}")
        
        # Αποθήκευση profiling
        cluster_profiles.to_csv('cluster_profiles.csv', index=False)
        print(f"\n✓ Cluster profiles αποθηκεύτηκαν στο cluster_profiles.csv")
        
        return cluster_profiles
    
    def _interpret_cluster(self, avg_length, avg_refs, avg_infobox):
        """Απλοποιημένη ερμηνεία cluster"""
        if avg_length > 10000 and avg_refs > 30:
            return "Εκτενή άρθρα με υψηλή τεκμηρίωση"
        elif avg_length < 3000 and avg_refs < 10:
            return "Σύντομα άρθρα περιορισμένης δομής"
        elif avg_infobox > 10:
            return "Άρθρα έντονης δομής με infobox"
        elif avg_refs > 20:
            return "Καλά τεκμηριωμένα άρθρα μεσαίου μήκους"
        else:
            return "Άρθρα γενικού περιεχομένου"
    
    def save_results(self, predictions):
        """
        Αποθήκευση τελικών αποτελεσμάτων
        
        Args:
            predictions: DataFrame με cluster assignments
        """
        print("\nΑποθήκευση αποτελεσμάτων...")
        
        # Επιλογή στηλών για export
        results = predictions.select(
            'url', 'title', 'cluster',
            'page_length_chars', 'num_sections', 'num_references',
            'num_categories', 'num_images', 'num_infobox_rows'
        )
        
        # Convert to Pandas και save
        results_pd = results.toPandas()
        results_pd.to_csv('clustering_results.csv', index=False)
        
        print(f"✓ Αποθηκεύτηκαν {len(results_pd)} εγγραφές στο clustering_results.csv")
        
        return results_pd
    
    def stop(self):
        """Σταμάτημα Spark session"""
        self.spark.stop()


def main_clustering_pipeline(jsonl_file: str = "wikipedia_features.jsonl",
                              use_scaling: bool = True,
                              k_range: range = range(2, 9)):
    """
    Πλήρης clustering pipeline
    
    Args:
        jsonl_file: Input JSONL file
        use_scaling: Χρήση StandardScaler (πειραματισμός 6)
        k_range: Εύρος k για αξιολόγηση
    """
    print("="*60)
    print("SPARK MLLIB K-MEANS CLUSTERING")
    print("="*60)
    print(f"Input file: {jsonl_file}")
    print(f"Use scaling: {use_scaling}")
    print(f"K range: {k_range.start}-{k_range.stop-1}")
    print("="*60)
    
    # Δημιουργία clusterer
    clusterer = WikipediaClusterer(use_scaling=use_scaling)
    
    # Βήμα 5: Φόρτωση δεδομένων
    clusterer.load_data(jsonl_file)
    
    # Καθαρισμός
    clusterer.clean_data()
    
    # Βήμα 6: Feature vector και scaling
    clusterer.create_features()
    
    # Βήμα 7: Εύρεση βέλτιστου k
    results, best_k = clusterer.find_optimal_k(k_range)
    
    # Τελικό clustering
    predictions, model = clusterer.final_clustering(best_k)
    
    # Profiling clusters
    cluster_profiles = clusterer.profile_clusters(predictions)
    
    # Αποθήκευση αποτελεσμάτων
    results_df = clusterer.save_results(predictions)
    
    # Σταμάτημα Spark
    clusterer.stop()
    
    print("\n✓ Clustering pipeline ολοκληρώθηκε επιτυχώς!")
    
    return results_df, cluster_profiles


if __name__ == "__main__":
    # Εκτέλεση με default παραμέτρους
    results_df, cluster_profiles = main_clustering_pipeline(
        jsonl_file="wikipedia_features.jsonl",
        use_scaling=True,
        k_range=range(2, 9)
    )