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
    
    label_column = 'AF'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'Data', 'shap_filtered_datasets', label_column)
    output_dir = os.path.join(script_dir, 'Results', f'clustering_metrics_results_datasets_{label_column}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to file and terminal
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'clustering_evaluation_datasets_log_{timestamp}.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    
    #datasets = ['chapman_ningbo', 'ptbxl', 'cod15', 'georgia']
    datasets = ['chapman_ningbo', 'cod15', 'georgia']
    models_conf = [model_conf_path for model_conf_path in os.listdir(os.path.join(base_path)) if os.path.isdir(os.path.join(base_path, model_conf_path))]
    
    # List to store all results
    all_results = []
    pairwise_results = []
    dataset_label_results = []
    pairwise_dataset_label_results = []
    
    print("="*80)
    print("DATASET SEPARATION ANALYSIS IN EMBEDDING SPACE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total model configurations: {len(models_conf)}")
    print(f"Datasets to analyze: {len(datasets)}")
    print(f"Analyzing: (1) Dataset separation, (2) Pairwise datasets, (3) Dataset+Label, (4) Pairwise Dataset+Label\n")
    
    for idx_model, model_conf in enumerate(models_conf, 1):
        print(f"\n{'='*80}")
        print(f"MODEL CONFIGURATION [{idx_model}/{len(models_conf)}]: {model_conf}")
        print(f"{'='*80}")
        
        # Load all datasets for this model configuration
        all_features = []
        all_dataset_labels = []
        all_cd_labels = []
        dataset_names = []
        
        print(f"\n  Loading all datasets...")
        for idx_dataset, dataset in enumerate(datasets):
            try:
                csv_path = os.path.join(base_path, f'{model_conf}/{dataset}_{label_column}_dataset.csv')
                datas = pd.read_csv(csv_path, header=0)
                
                feature_col = [col for col in datas.columns if 'feature_' in col]
                features = datas[feature_col].to_numpy()
                cd_labels = datas[label_column].to_numpy()
                
                all_features.append(features)
                all_dataset_labels.extend([idx_dataset] * len(features))
                all_cd_labels.extend(cd_labels)
                dataset_names.append(dataset)
                
                print(f"    ✓ Loaded {dataset}: {len(features)} samples, {len(feature_col)} features")
                
            except Exception as e:
                print(f"    ✗ Error loading {dataset}: {str(e)}")
                continue
        
        if len(all_features) == 0:
            print(f"  ✗ No datasets loaded for {model_conf}, skipping...")
            continue
        
        # Concatenate all features
        combined_features = np.vstack(all_features)
        dataset_labels = np.array(all_dataset_labels)
        cd_labels = np.array(all_cd_labels)
        
        total_samples = len(combined_features)
        print(f"\n  Combined: {total_samples} total samples from {len(dataset_names)} datasets")
        
        # =================================================================
        # PART 1: OVERALL DATASET SEPARATION
        # =================================================================
        print(f"\n  {'='*70}")
        print(f"  [1/3] ANALYZING OVERALL DATASET SEPARATION")
        print(f"  {'='*70}")
        
        result = {
            'model_configuration': model_conf,
            'n_datasets': len(dataset_names),
            'total_samples': total_samples
        }
        
        # Dataset distribution
        for idx, dataset_name in enumerate(dataset_names):
            count = np.sum(dataset_labels == idx)
            print(f"    • {dataset_name}: {count} samples ({count/total_samples*100:.1f}%)")
            result[f'n_samples_{dataset_name}'] = count
        
        # Standard clustering quality metrics (datasets as clusters)
        print(f"\n    Computing clustering quality metrics (datasets as clusters)...")
        try:
            quality = evaluate_clusters_quality_standard(
                combined_features, dataset_labels,
                use_scaler=True,
                distance_metric='euclidean'
            )
            result.update({
                'overall_silhouette_score': quality['silhouette_score'],
                'overall_davies_bouldin_score': quality['davies_bouldin_score'],
                'overall_calinski_harabasz_score': quality['calinski_harabasz_score']
            })
            print(f"      Silhouette Score: {quality['silhouette_score']:.4f}")
            print(f"      Davies-Bouldin Score: {quality['davies_bouldin_score']:.4f}")
            print(f"      Calinski-Harabasz Score: {quality['calinski_harabasz_score']:.2f}")
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            result.update({
                'overall_silhouette_score': np.nan,
                'overall_davies_bouldin_score': np.nan,
                'overall_calinski_harabasz_score': np.nan
            })
        
        # KNN dataset agreement
        print(f"\n    Computing KNN dataset agreement...")
        try:
            knn_agreement = knn_label_agreement(
                combined_features, dataset_labels,
                k=10,
                use_scaler=True,
                distance_metric='euclidean'
            )
            result['overall_knn_dataset_agreement'] = knn_agreement
            print(f"      KNN Dataset Agreement (k=10): {knn_agreement:.4f}")
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            result['overall_knn_dataset_agreement'] = np.nan
        
        all_results.append(result)
        
        # =================================================================
        # PART 2: PAIRWISE DATASET COMPARISONS
        # =================================================================
        print(f"\n  {'='*70}")
        print(f"  [2/3] PAIRWISE DATASET SEPARATION ANALYSIS")
        print(f"  {'='*70}")
        
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                dataset_i = dataset_names[i]
                dataset_j = dataset_names[j]
                
                print(f"\n    Comparing: {dataset_i.upper()} vs {dataset_j.upper()}")
                
                # Extract features for these two datasets
                mask = (dataset_labels == i) | (dataset_labels == j)
                pair_features = combined_features[mask]
                pair_labels = dataset_labels[mask]
                
                # Convert to binary labels (0 and 1)
                pair_labels_binary = (pair_labels == j).astype(int)
                
                pair_result = {
                    'model_configuration': model_conf,
                    'dataset_1': dataset_i,
                    'dataset_2': dataset_j,
                    'n_samples_1': np.sum(pair_labels == i),
                    'n_samples_2': np.sum(pair_labels == j)
                }
                
                # Silhouette score for pair
                try:
                    X_scaled = StandardScaler().fit_transform(pair_features)
                    silhouette = silhouette_score(X_scaled, pair_labels_binary, metric='euclidean')
                    pair_result['pairwise_silhouette'] = silhouette
                    print(f"      Silhouette Score: {silhouette:.4f}")
                except Exception as e:
                    print(f"      ✗ Silhouette Error: {str(e)}")
                    pair_result['pairwise_silhouette'] = np.nan
                
                # Centroid separation for pair
                try:
                    separation = evaluate_centroid_separation(
                        pair_features, pair_labels_binary,
                        use_scaler=True,
                        whiten=True,
                        distance_metric='euclidean'
                    )
                    pair_result.update({
                        'pairwise_centroid_distance': separation['centroid_distance'],
                        'pairwise_centroid_separation': separation['centroid_separation']
                    })
                    print(f"      Centroid Distance: {separation['centroid_distance']:.4f}")
                    print(f"      Centroid Separation: {separation['centroid_separation']:.4f}")
                except Exception as e:
                    print(f"      ✗ Centroid Error: {str(e)}")
                    pair_result.update({
                        'pairwise_centroid_distance': np.nan,
                        'pairwise_centroid_separation': np.nan
                    })
                
                # KNN agreement for pair
                try:
                    knn_agr = knn_label_agreement(
                        pair_features, pair_labels_binary,
                        k=10,
                        use_scaler=True,
                        distance_metric='euclidean'
                    )
                    pair_result['pairwise_knn_agreement'] = knn_agr
                    print(f"      KNN Agreement: {knn_agr:.4f}")
                except Exception as e:
                    print(f"      ✗ KNN Error: {str(e)}")
                    pair_result['pairwise_knn_agreement'] = np.nan
                
                # Unsupervised recovery for pair
                try:
                    recovery = evaluate_unsupervised_recovery(
                        pair_features, pair_labels_binary,
                        use_scaler=True,
                        spectral_n_neighbors=10
                    )
                    pair_result.update({
                        'pairwise_spectral_ari': recovery['Spectral_ARI'],
                        'pairwise_spectral_ami': recovery['Spectral_AMI'],
                        'pairwise_gmm_ari': recovery['GMM_ARI'],
                        'pairwise_gmm_ami': recovery['GMM_AMI']
                    })
                    print(f"      Spectral ARI: {recovery['Spectral_ARI']:.4f}, AMI: {recovery['Spectral_AMI']:.4f}")
                    print(f"      GMM ARI: {recovery['GMM_ARI']:.4f}, AMI: {recovery['GMM_AMI']:.4f}")
                except Exception as e:
                    print(f"      ✗ Unsupervised Recovery Error: {str(e)}")
                    pair_result.update({
                        'pairwise_spectral_ari': np.nan,
                        'pairwise_spectral_ami': np.nan,
                        'pairwise_gmm_ari': np.nan,
                        'pairwise_gmm_ami': np.nan
                    })
                
                pairwise_results.append(pair_result)
        
        # =================================================================
        # PART 3: DATASET + CD LABEL COMBINATIONS (8 clusters)
        # =================================================================
        print(f"\n  {'='*70}")
        print(f"  [3/3] DATASET+LABEL COMBINATION ANALYSIS")
        print(f"  {'='*70}")
        
        # Create combined labels: dataset_idx * 2 + cd_label
        combined_labels = dataset_labels * 2 + cd_labels
        
        print(f"\n    Analyzing {len(dataset_names) * 2} clusters (dataset × CD label):")
        for idx, dataset_name in enumerate(dataset_names):
            for cd in [0, 1]:
                cluster_id = idx * 2 + cd
                count = np.sum(combined_labels == cluster_id)
                print(f"      • {dataset_name}_CD{cd}: {count} samples ({count/total_samples*100:.1f}%)")
        
        dl_result = {
            'model_configuration': model_conf,
            'n_clusters': len(dataset_names) * 2
        }
        
        # Clustering quality for dataset+label combinations
        print(f"\n    Computing clustering metrics for dataset+label combinations...")
        try:
            quality_dl = evaluate_clusters_quality_standard(
                combined_features, combined_labels,
                use_scaler=True,
                distance_metric='euclidean'
            )
            dl_result.update({
                'dataset_label_silhouette': quality_dl['silhouette_score'],
                'dataset_label_davies_bouldin': quality_dl['davies_bouldin_score'],
                'dataset_label_calinski_harabasz': quality_dl['calinski_harabasz_score']
            })
            print(f"      Silhouette Score: {quality_dl['silhouette_score']:.4f}")
            print(f"      Davies-Bouldin Score: {quality_dl['davies_bouldin_score']:.4f}")
            print(f"      Calinski-Harabasz Score: {quality_dl['calinski_harabasz_score']:.2f}")
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            dl_result.update({
                'dataset_label_silhouette': np.nan,
                'dataset_label_davies_bouldin': np.nan,
                'dataset_label_calinski_harabasz': np.nan
            })
        
        # KNN agreement for dataset+label
        try:
            knn_agr_dl = knn_label_agreement(
                combined_features, combined_labels,
                k=10,
                use_scaler=True,
                distance_metric='euclidean'
            )
            dl_result['dataset_label_knn_agreement'] = knn_agr_dl
            print(f"      KNN Agreement: {knn_agr_dl:.4f}")
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            dl_result['dataset_label_knn_agreement'] = np.nan
        
        dataset_label_results.append(dl_result)
        
        # =================================================================
        # PART 4: PAIRWISE DATASET+LABEL COMPARISONS
        # =================================================================
        print(f"\n  {'='*70}")
        print(f"  [4/4] PAIRWISE DATASET+LABEL COMBINATION ANALYSIS")
        print(f"  {'='*70}")
        
        # Create list of all dataset+label combinations
        dataset_label_combos = []
        for idx, dataset_name in enumerate(dataset_names):
            for cd in [0, 1]:
                dataset_label_combos.append({
                    'name': f"{dataset_name}_CD{cd}",
                    'dataset_idx': idx,
                    'cd_label': cd,
                    'cluster_id': idx * 2 + cd
                })
        
        n_combos = len(dataset_label_combos)
        n_comparisons = (n_combos * (n_combos - 1)) // 2
        print(f"\n    Total combinations: {n_combos}")
        print(f"    Total pairwise comparisons: {n_comparisons}\n")
        
        comparison_count = 0
        for i in range(len(dataset_label_combos)):
            for j in range(i + 1, len(dataset_label_combos)):
                comparison_count += 1
                combo_i = dataset_label_combos[i]
                combo_j = dataset_label_combos[j]
                
                print(f"\n    [{comparison_count}/{n_comparisons}] Comparing: {combo_i['name']} vs {combo_j['name']}")
                
                # Extract features for these two combinations
                mask_i = combined_labels == combo_i['cluster_id']
                mask_j = combined_labels == combo_j['cluster_id']
                mask = mask_i | mask_j
                
                pair_features = combined_features[mask]
                pair_labels = combined_labels[mask]
                
                # Convert to binary labels (0 and 1)
                pair_labels_binary = (pair_labels == combo_j['cluster_id']).astype(int)
                
                n_samples_i = np.sum(mask_i)
                n_samples_j = np.sum(mask_j)
                
                pair_dl_result = {
                    'model_configuration': model_conf,
                    'combination_1': combo_i['name'],
                    'combination_2': combo_j['name'],
                    'dataset_1': dataset_names[combo_i['dataset_idx']],
                    'cd_label_1': combo_i['cd_label'],
                    'dataset_2': dataset_names[combo_j['dataset_idx']],
                    'cd_label_2': combo_j['cd_label'],
                    'n_samples_1': n_samples_i,
                    'n_samples_2': n_samples_j,
                    'same_dataset': combo_i['dataset_idx'] == combo_j['dataset_idx'],
                    'same_cd_label': combo_i['cd_label'] == combo_j['cd_label']
                }
                
                print(f"      Samples: {n_samples_i} vs {n_samples_j}")
                print(f"      Same dataset: {combo_i['dataset_idx'] == combo_j['dataset_idx']}, Same CD label: {combo_i['cd_label'] == combo_j['cd_label']}")
                
                # Silhouette score for pair
                try:
                    X_scaled = StandardScaler().fit_transform(pair_features)
                    silhouette = silhouette_score(X_scaled, pair_labels_binary, metric='euclidean')
                    pair_dl_result['pairwise_dl_silhouette'] = silhouette
                    print(f"      Silhouette Score: {silhouette:.4f}")
                except Exception as e:
                    print(f"      ✗ Silhouette Error: {str(e)}")
                    pair_dl_result['pairwise_dl_silhouette'] = np.nan
                
                # Centroid separation for pair
                try:
                    separation = evaluate_centroid_separation(
                        pair_features, pair_labels_binary,
                        use_scaler=True,
                        whiten=True,
                        distance_metric='euclidean'
                    )
                    pair_dl_result.update({
                        'pairwise_dl_centroid_distance': separation['centroid_distance'],
                        'pairwise_dl_centroid_separation': separation['centroid_separation'],
                        'pairwise_dl_intra_class_dispersion': separation['intra_class_dispersion']
                    })
                    print(f"      Centroid Distance: {separation['centroid_distance']:.4f}")
                    print(f"      Centroid Separation: {separation['centroid_separation']:.4f}")
                except Exception as e:
                    print(f"      ✗ Centroid Error: {str(e)}")
                    pair_dl_result.update({
                        'pairwise_dl_centroid_distance': np.nan,
                        'pairwise_dl_centroid_separation': np.nan,
                        'pairwise_dl_intra_class_dispersion': np.nan
                    })
                
                # KNN agreement for pair
                try:
                    knn_agr = knn_label_agreement(
                        pair_features, pair_labels_binary,
                        k=10,
                        use_scaler=True,
                        distance_metric='euclidean'
                    )
                    pair_dl_result['pairwise_dl_knn_agreement'] = knn_agr
                    print(f"      KNN Agreement: {knn_agr:.4f}")
                except Exception as e:
                    print(f"      ✗ KNN Error: {str(e)}")
                    pair_dl_result['pairwise_dl_knn_agreement'] = np.nan
                
                # Unsupervised recovery for pair
                try:
                    recovery = evaluate_unsupervised_recovery(
                        pair_features, pair_labels_binary,
                        use_scaler=True,
                        spectral_n_neighbors=10
                    )
                    pair_dl_result.update({
                        'pairwise_dl_spectral_ari': recovery['Spectral_ARI'],
                        'pairwise_dl_spectral_ami': recovery['Spectral_AMI'],
                        'pairwise_dl_gmm_ari': recovery['GMM_ARI'],
                        'pairwise_dl_gmm_ami': recovery['GMM_AMI']
                    })
                    print(f"      Spectral ARI: {recovery['Spectral_ARI']:.4f}, AMI: {recovery['Spectral_AMI']:.4f}")
                    print(f"      GMM ARI: {recovery['GMM_ARI']:.4f}, AMI: {recovery['GMM_AMI']:.4f}")
                except Exception as e:
                    print(f"      ✗ Unsupervised Recovery Error: {str(e)}")
                    pair_dl_result.update({
                        'pairwise_dl_spectral_ari': np.nan,
                        'pairwise_dl_spectral_ami': np.nan,
                        'pairwise_dl_gmm_ari': np.nan,
                        'pairwise_dl_gmm_ami': np.nan
                    })
                
                pairwise_dataset_label_results.append(pair_dl_result)
        
        print(f"    ✓ Completed {n_comparisons} pairwise dataset+label comparisons")
        
        print(f"\n  ✓ Completed analysis for {model_conf}")
    
    # =================================================================
    # SAVE OVERALL AND PAIRWISE DATASET SEPARATION RESULTS
    # =================================================================
    print(f"\n{'='*80}")
    print("SAVING DATASET SEPARATION RESULTS (DATASETS AS CLUSTERS, NO LABEL SPLIT)")
    print(f"{'='*80}\n")

    overall_df = pd.DataFrame(all_results)
    overall_output = os.path.join(output_dir, 'dataset_overall_separation.csv')
    overall_df.to_csv(overall_output, index=False)
    print(f"✓ Overall dataset separation saved to: {overall_output}")

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_output = os.path.join(output_dir, 'dataset_pairwise_comparisons.csv')
    pairwise_df.to_csv(pairwise_output, index=False)
    print(f"✓ Pairwise dataset comparisons saved to: {pairwise_output}")

    # =================================================================
    # SAVE FINAL RESULTS - CONFUSION MATRIX STYLE TABLES ONLY
    # =================================================================
    print(f"\n{'='*80}")
    print("GENERATING CONFUSION MATRIX STYLE TABLES (DATASET+LABEL COMPARISONS)")
    print(f"{'='*80}\n")

    # Convert results to DataFrame
    pairwise_dl_df = pd.DataFrame(pairwise_dataset_label_results)
    
    if len(pairwise_dl_df) > 0:
        
        matrix_dir = os.path.join(output_dir, 'confusion_matrix_style')
        os.makedirs(matrix_dir, exist_ok=True)
        
        key_metrics_matrix = [
            'pairwise_dl_knn_agreement',
            'pairwise_dl_centroid_distance',
            'pairwise_dl_centroid_separation',
            'pairwise_dl_spectral_ari',
            'pairwise_dl_gmm_ari'
        ]
        
        # For each model configuration
        for model_conf in pairwise_dl_df['model_configuration'].unique():
            model_data = pairwise_dl_df[pairwise_dl_df['model_configuration'] == model_conf]
            
            # Get all unique combinations
            all_combos = sorted(set(model_data['combination_1'].unique()) | set(model_data['combination_2'].unique()))
            
            # For each metric, create a confusion matrix
            for metric in key_metrics_matrix:
                if metric not in model_data.columns:
                    continue
                
                # Initialize matrix with NaN, diagonal set to 1.0 or 0.0 depending on metric type
                diag_value = 1.0 if ('ari' in metric or 'knn' in metric) else 0.0
                matrix_values = np.full((len(all_combos), len(all_combos)), np.nan)
                np.fill_diagonal(matrix_values, diag_value)
                matrix = pd.DataFrame(matrix_values, index=all_combos, columns=all_combos)
                
                # Fill the matrix with pairwise values
                for _, row in model_data.iterrows():
                    combo1 = row['combination_1']
                    combo2 = row['combination_2']
                    value = row[metric]
                    
                    # Fill both directions (symmetric)
                    matrix.loc[combo1, combo2] = value
                    matrix.loc[combo2, combo1] = value
                
                # Save matrix
                metric_name = metric.replace('pairwise_dl_', '')
                matrix_file = os.path.join(matrix_dir, f'{model_conf}_{metric_name}_matrix.csv')
                matrix.to_csv(matrix_file)
                
            print(f"  ✓ Saved {len(key_metrics_matrix)} matrices for {model_conf}")
        
        print(f"\n✓ All confusion matrix style tables saved to: {matrix_dir}")
    
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nTotal model configurations analyzed: {len(pairwise_dl_df['model_configuration'].unique())}")
    print(f"Total pairwise dataset+label comparisons: {len(pairwise_dl_df)}")
    print(f"\nConfusion matrix style tables saved to: {output_dir}/confusion_matrix_style/")
    print(f"Log file saved to: {log_file}")
    print(f"{'='*80}\n")
    
    # Close logger
    logger.close()
    sys.stdout = logger.terminal
    
    