import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from datetime import datetime


class Logger:
    """Class to log output both to terminal and to file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def evaluate_clusters_quality_standard(features, labels, use_scaler, distance_metric):
    
    if use_scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    silhouette_avg = silhouette_score(features, labels, metric=distance_metric)
    davies_bouldin_avg = davies_bouldin_score(features, labels)
    calinski_harabasz_avg = calinski_harabasz_score(features, labels)
    
    metrics = {
        'silhouette_score': silhouette_avg,
        'davies_bouldin_score': davies_bouldin_avg,
        'calinski_harabasz_score': calinski_harabasz_avg
    }

    return metrics

def knn_label_agreement(features, labels, k=10, use_scaler=True, distance_metric="euclidean"):
    """
    features : np.ndarray (N, D)
    labels   : np.ndarray (N,)
    """

    X = features.copy()

    if use_scaler:
        X = StandardScaler().fit_transform(X)

    nn = NearestNeighbors(
        n_neighbors=k + 1,   # +1 perché include se stesso
        metric=distance_metric
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    agreements = []

    for i in range(len(labels)):
        neighbors_idx = indices[i][1:]  # escludi se stesso
        same_label = np.mean(labels[neighbors_idx] == labels[i])
        agreements.append(same_label)

    return float(np.mean(agreements))


def evaluate_unsupervised_recovery(features, labels, use_scaler=True,spectral_n_neighbors=10, random_state=42):
    """
    features : np.ndarray (N, D)
    labels   : np.ndarray (N,)  binary {0,1}

    Returns:
        dict with ARI / AMI for Spectral Clustering and GMM
    """

    X = features.copy()

    # -----------------------
    # 1. Scaling
    # -----------------------
    if use_scaler:
        X = StandardScaler().fit_transform(X)
 
    # -----------------------
    # 3. Spectral Clustering
    # -----------------------
    spectral = SpectralClustering(
        n_clusters=2,
        affinity="nearest_neighbors",
        n_neighbors=spectral_n_neighbors,
        assign_labels="kmeans",
        random_state=random_state
    )

    spectral_labels = spectral.fit_predict(X)

    spectral_ari = adjusted_rand_score(labels, spectral_labels)
    spectral_ami = adjusted_mutual_info_score(labels, spectral_labels)

    # -----------------------
    # 4. Gaussian Mixture Model
    # -----------------------
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=random_state
    )

    gmm_labels = gmm.fit_predict(X)

    gmm_ari = adjusted_rand_score(labels, gmm_labels)
    gmm_ami = adjusted_mutual_info_score(labels, gmm_labels)

    # -----------------------
    # 5. Output
    # -----------------------
    results = {
        "Spectral_ARI": spectral_ari,
        "Spectral_AMI": spectral_ami,
        "GMM_ARI": gmm_ari,
        "GMM_AMI": gmm_ami
    }

    return results

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


def evaluate_centroid_separation(
    features,
    labels,
    use_scaler=True,
    whiten=True,
    distance_metric="euclidean"
):
    """
    features : np.ndarray (N, D)
    labels   : np.ndarray (N,) with values {0,1}

    Returns:
        dict with centroid distance and normalized separation
    """

    X = features.copy()

    # -----------------------
    # 1. Scaling
    # -----------------------
    if use_scaler:
        X = StandardScaler().fit_transform(X)

    # -----------------------
    # 3. Split by class
    # -----------------------
    X0 = X[labels == 0]
    X1 = X[labels == 1]

    if len(X0) == 0 or len(X1) == 0:
        raise ValueError("One of the classes is empty.")

    # -----------------------
    # 4. Centroids
    # -----------------------
    mu0 = X0.mean(axis=0, keepdims=True)
    mu1 = X1.mean(axis=0, keepdims=True)

    # -----------------------
    # 5. Distances
    # -----------------------
    centroid_distance = cdist(mu0, mu1, metric=distance_metric)[0, 0]

    # intra-class dispersion
    sigma0 = cdist(X0, mu0, metric=distance_metric).mean()
    sigma1 = cdist(X1, mu1, metric=distance_metric).mean()
    intra_class_dispersion = 0.5 * (sigma0 + sigma1)

    # normalized separation (effect-size)
    centroid_separation = centroid_distance / intra_class_dispersion

    return {
        "centroid_distance": float(centroid_distance),
        "intra_class_dispersion": float(intra_class_dispersion),
        "centroid_separation": float(centroid_separation)
    }

if __name__ == '__main__':

    label_column = 'CD'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'Data', 'shap_filtered_datasets', label_column)
    output_dir = os.path.join(script_dir, 'Results', f'clustering_metrics_results_labels_{label_column}')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging to file and terminal
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'clustering_evaluation_log_{timestamp}.txt')
    logger = Logger(log_file)
    sys.stdout = logger

    datasets = ['chapman_ningbo', 'ptbxl', 'cod15', 'georgia']
    models_conf = sorted(model_conf_path for model_conf_path in os.listdir(base_path)
                          if os.path.isdir(os.path.join(base_path, model_conf_path)))

    # List to store all results
    all_results = []
    
    print("="*80)
    print("CLUSTERING METRICS EVALUATION FOR EMBEDDING QUALITY ASSESSMENT")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total configurations: {len(models_conf)}")
    print(f"Datasets per configuration: {len(datasets)}")
    print(f"Total evaluations: {len(models_conf) * len(datasets)}\n")
    
    for idx_model, model_conf in enumerate(models_conf, 1):
        print(f"\n{'='*80}")
        print(f"MODEL CONFIGURATION [{idx_model}/{len(models_conf)}]: {model_conf}")
        print(f"{'='*80}")
        
        for idx_dataset, dataset in enumerate(datasets, 1):
            print(f"\n  [{idx_dataset}/{len(datasets)}] Processing dataset: {dataset.upper()}")
            print(f"  {'-'*70}")
            
            try:
                # Load data
                csv_path = os.path.join(base_path, f'{model_conf}/{dataset}_{label_column}_dataset.csv')
                datas = pd.read_csv(csv_path, header=0)

                feature_col = [col for col in datas.columns if 'feature_' in col]
                features = datas[feature_col].to_numpy()
                labels = datas[label_column].to_numpy()
                
                n_samples = len(labels)
                n_features = len(feature_col)
                n_class_0 = np.sum(labels == 0)
                n_class_1 = np.sum(labels == 1)
                
                print(f"    • Samples: {n_samples} | Features: {n_features}")
                print(f"    • Class distribution: {label_column}=0: {n_class_0} ({n_class_0/n_samples*100:.1f}%), {label_column}=1: {n_class_1} ({n_class_1/n_samples*100:.1f}%)")
                
                # Initialize result dictionary
                result = {
                    'model_configuration': model_conf,
                    'dataset': dataset,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'n_class_0': n_class_0,
                    'n_class_1': n_class_1,
                    'class_balance_ratio': min(n_class_0, n_class_1) / max(n_class_0, n_class_1)
                }
                
                # 1. Standard clustering quality metrics
                print(f"\n    [1/4] Computing standard clustering quality metrics...")
                try:
                    quality = evaluate_clusters_quality_standard(
                        features, labels, 
                        use_scaler=True, 
                        distance_metric='euclidean'
                    )
                    result.update({
                        'silhouette_score': quality['silhouette_score'],
                        'davies_bouldin_score': quality['davies_bouldin_score'],
                        'calinski_harabasz_score': quality['calinski_harabasz_score']
                    })
                    print(f"          Silhouette Score: {quality['silhouette_score']:.4f}")
                    print(f"          Davies-Bouldin Score: {quality['davies_bouldin_score']:.4f}")
                    print(f"          Calinski-Harabasz Score: {quality['calinski_harabasz_score']:.2f}")
                except Exception as e:
                    print(f"          ✗ Error: {str(e)}")
                    result.update({
                        'silhouette_score': np.nan,
                        'davies_bouldin_score': np.nan,
                        'calinski_harabasz_score': np.nan
                    })
                
                # 2. KNN label agreement
                print(f"\n    [2/4] Computing KNN label agreement...")
                try:
                    knn_agreement = knn_label_agreement(
                        features, labels, 
                        k=10, 
                        use_scaler=True, 
                        distance_metric='euclidean'
                    )
                    result['knn_label_agreement_k10'] = knn_agreement
                    print(f"          KNN Label Agreement (k=10): {knn_agreement:.4f}")
                except Exception as e:
                    print(f"          ✗ Error: {str(e)}")
                    result['knn_label_agreement_k10'] = np.nan
                
                # 3. Unsupervised recovery metrics
                print(f"\n    [3/4] Computing unsupervised recovery metrics...")
                try:
                    recovery = evaluate_unsupervised_recovery(
                        features, labels, 
                        use_scaler=True, 
                        spectral_n_neighbors=10
                    )
                    result.update({
                        'spectral_ari': recovery['Spectral_ARI'],
                        'spectral_ami': recovery['Spectral_AMI'],
                        'gmm_ari': recovery['GMM_ARI'],
                        'gmm_ami': recovery['GMM_AMI']
                    })
                    print(f"          Spectral Clustering - ARI: {recovery['Spectral_ARI']:.4f}, AMI: {recovery['Spectral_AMI']:.4f}")
                    print(f"          Gaussian Mixture Model - ARI: {recovery['GMM_ARI']:.4f}, AMI: {recovery['GMM_AMI']:.4f}")
                except Exception as e:
                    print(f"          ✗ Error: {str(e)}")
                    result.update({
                        'spectral_ari': np.nan,
                        'spectral_ami': np.nan,
                        'gmm_ari': np.nan,
                        'gmm_ami': np.nan
                    })
                
                # 4. Centroid separation metrics
                print(f"\n    [4/4] Computing centroid separation metrics...")
                try:
                    separation = evaluate_centroid_separation(
                        features, labels, 
                        use_scaler=True, 
                        whiten=True, 
                        distance_metric='euclidean'
                    )
                    result.update({
                        'centroid_distance': separation['centroid_distance'],
                        'intra_class_dispersion': separation['intra_class_dispersion'],
                        'centroid_separation': separation['centroid_separation']
                    })
                    print(f"          Centroid Distance: {separation['centroid_distance']:.4f}")
                    print(f"          Intra-class Dispersion: {separation['intra_class_dispersion']:.4f}")
                    print(f"          Centroid Separation (normalized): {separation['centroid_separation']:.4f}")
                except Exception as e:
                    print(f"          ✗ Error: {str(e)}")
                    result.update({
                        'centroid_distance': np.nan,
                        'intra_class_dispersion': np.nan,
                        'centroid_separation': np.nan
                    })
                
                all_results.append(result)
                
                # Save intermediate results after each dataset
                if len(all_results) > 0:
                    temp_df = pd.DataFrame(all_results)
                    temp_output = os.path.join(output_dir, 'clustering_metrics_partial.csv')
                    temp_df.to_csv(temp_output, index=False)
                
                print(f"\n  ✓ Successfully processed {dataset}")
                
            except Exception as e:
                print(f"\n  ✗ Error processing {dataset}: {str(e)}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    detailed_output = os.path.join(output_dir, 'clustering_metrics_detailed.csv')
    results_df.to_csv(detailed_output, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Detailed results saved to: {detailed_output}")
    
    # Save summary statistics by model
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY STATISTICS BY MODEL CONFIGURATION")
    print(f"{'='*80}\n")
    
    metric_columns = [
        'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score',
        'knn_label_agreement_k10', 'spectral_ari', 'spectral_ami', 
        'gmm_ari', 'gmm_ami', 'centroid_distance', 
        'intra_class_dispersion', 'centroid_separation'
    ]
    
    summary_by_model = results_df.groupby('model_configuration')[metric_columns].agg(['mean', 'std'])
    summary_by_model.columns = ['_'.join(col).strip() for col in summary_by_model.columns.values]
    summary_by_model = summary_by_model.reset_index()
    
    summary_model_output = os.path.join(output_dir, 'clustering_metrics_summary_by_model.csv')
    summary_by_model.to_csv(summary_model_output, index=False)
    print(f"✓ Summary by model saved to: {summary_model_output}")
    
    # Save summary statistics by dataset
    summary_by_dataset = results_df.groupby('dataset')[metric_columns].agg(['mean', 'std'])
    summary_by_dataset.columns = ['_'.join(col).strip() for col in summary_by_dataset.columns.values]
    summary_by_dataset = summary_by_dataset.reset_index()
    
    summary_dataset_output = os.path.join(output_dir, 'clustering_metrics_summary_by_dataset.csv')
    summary_by_dataset.to_csv(summary_dataset_output, index=False)
    print(f"✓ Summary by dataset saved to: {summary_dataset_output}")
    
    # Create a pivot table for easy comparison (model x dataset)
    print(f"\n{'='*80}")
    print("GENERATING PIVOT TABLES FOR KEY METRICS")
    print(f"{'='*80}\n")
    
    key_metrics = ['silhouette_score', 'knn_label_agreement_k10', 'spectral_ari', 'centroid_separation']
    
    for metric in key_metrics:
        pivot = results_df.pivot(index='model_configuration', columns='dataset', values=metric)
        pivot_output = os.path.join(output_dir, f'pivot_{metric}.csv')
        pivot.to_csv(pivot_output)
        print(f"✓ Pivot table for '{metric}' saved to: {pivot_output}")
    
    
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nTotal configurations evaluated: {len(results_df['model_configuration'].unique())}")
    print(f"Total datasets evaluated: {len(results_df['dataset'].unique())}")
    print(f"Total evaluations: {len(results_df)}")
    print(f"\nAll results have been saved to: {output_dir}")
    print(f"Log file saved to: {log_file}")
    print(f"{'='*80}\n")
    
    # Close logger
    logger.close()
    sys.stdout = logger.terminal
    
    