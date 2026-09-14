# Cluster Embeddings Analysis

Toolkit di analisi per valutare **quanto sono ben separati/organizzati embedding
(vettori di features numeriche)** rispetto a delle etichette note — sia etichette
di classe (es. presenza/assenza di una patologia) sia etichette di provenienza
(es. il dataset da cui proviene ciascun campione).

L'idea di fondo: se un embedding è "buono", campioni con la stessa etichetta
dovrebbero stare vicini nello spazio delle feature e campioni con etichette
diverse dovrebbero stare separati. Gli script calcolano un insieme di metriche
di clustering/separazione per quantificare questo fenomeno, senza addestrare
alcun classificatore supervisionato — l'unico uso delle etichette è per valutare
la struttura geometrica già presente nello spazio degli embedding.

Il repository contiene due script indipendenti ma che condividono le stesse
funzioni di calcolo delle metriche:

| Script | Cosa confronta |
|---|---|
| [`calculate_clusters_embedding_labels.py`](calculate_clusters_embedding_labels.py) | Per ogni dataset, separazione tra le due classi di una label binaria (es. `CD` = 0/1) |
| [`calculate_clusters_embedding_dataset.py`](calculate_clusters_embedding_dataset.py) | Separazione tra dataset diversi (dataset come "cluster"), e combinazioni dataset×label |

---

## 1. Input atteso

### 1.1 Formato dei file

Entrambi gli script si aspettano dei file **CSV**, uno per ogni combinazione
`(configurazione modello, dataset)`, con questa struttura di colonne:

- Una o più colonne il cui nome **contiene la sottostringa `feature_`**
  (es. `feature_0`, `feature_1`, ..., `feature_767`): sono le componenti
  del vettore di embedding di ciascun campione (riga). Vengono estratte con:
  ```python
  feature_col = [col for col in datas.columns if 'feature_' in col]
  features = datas[feature_col].to_numpy()   # shape (N, D)
  ```
- Una colonna di **etichetta binaria** (`{0, 1}`), il cui nome dipende dallo
  script (vedi sotto), usata come ground truth per valutare la separazione.

Ogni riga del CSV = un campione (osservazione) = un embedding + la sua etichetta.

### 1.2 Organizzazione delle cartelle e path

Entrambi gli script leggono i CSV da una struttura a cartelle
`base_path/<model_conf>/<file>.csv`, dove ogni sottocartella di `base_path`
rappresenta **una configurazione di modello/pipeline** (es. un particolare
foundation model + tecnica di feature selection, tipo
`CD_COD15_ECG-FM_nn_10perc_lst`).

**`calculate_clusters_embedding_labels.py`**
```
base_path = '/home/edofroses/ecg_fm/shap_filtered_datasets'
```
- Legge, per ogni `model_conf` in una lista fissa (`models_conf`) e per ogni
  `dataset` in `['chapman_ningbo', 'ptbxl', 'cod15', 'georgia']`, il file:
  ```
  {base_path}/{model_conf}/{dataset}_CD_dataset.csv
  ```
- Colonna etichetta attesa: **`CD`** (binaria, 0/1).

**`calculate_clusters_embedding_dataset.py`**
```
label_column = 'AF'
base_path = f'/home/edofroses/ecg_fm/shap_filtered_datasets/{label_column}'
```
- `models_conf` viene **scoperto automaticamente** elencando le sottocartelle
  di `base_path`.
- Per ogni `model_conf` e per ogni `dataset` in
  `['chapman_ningbo', 'cod15', 'georgia']`, legge il file:
  ```
  {base_path}/{model_conf}/{dataset}_{label_column}_dataset.csv
  ```
- Colonna etichetta attesa: il nome contenuto in `label_column` (es. `AF`,
  binaria 0/1) — usata sia come label di classe sia (implicitamente) come
  "sotto-etichetta" all'interno di ogni dataset per l'analisi dataset+label.

> Nota: i path (`base_path`, `output_dir`, la lista `datasets`, la lista/
> scoperta di `models_conf`, il nome della label) sono **hard-coded** nel blocco
> `if __name__ == '__main__':` di ciascun file. Per usare questi script sui
> propri dati vanno modificati direttamente questi valori in testa al blocco
> main.

### 1.3 Struttura dati in memoria

Durante l'esecuzione i dati vengono rappresentati come:

- `features`: `np.ndarray` di forma `(N, D)` — `N` campioni, `D` dimensioni
  dell'embedding.
- `labels`: `np.ndarray` di forma `(N,)`, valori binari `{0, 1}` (etichetta di
  classe, o indice del dataset di provenienza, o combinazione dei due).

---

## 2. Metriche calcolate

Tutte le metriche condivise sono definite come funzioni riutilizzabili in cima
a ciascuno script (duplicate identicamente nei due file):

| Funzione | Metriche prodotte | Interpretazione |
|---|---|---|
| `evaluate_clusters_quality_standard` | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` | Qualità "standard" del clustering indotto dalle etichette: quanto i gruppi sono compatti e ben separati (scikit-learn) |
| `knn_label_agreement` | `knn_label_agreement` (k=10 di default) | Frazione media dei k vicini più prossimi di ciascun punto che condividono la stessa etichetta: misura la "purezza locale" del vicinato |
| `evaluate_unsupervised_recovery` | `Spectral_ARI`, `Spectral_AMI`, `GMM_ARI`, `GMM_AMI` | Esegue clustering **non supervisionato** (Spectral Clustering e Gaussian Mixture Model, 2 componenti) sui soli embedding e confronta i cluster trovati con le etichette vere tramite Adjusted Rand Index / Adjusted Mutual Information: misura quanto le due classi sono "naturalmente" separabili senza usare le label |
| `evaluate_centroid_separation` | `centroid_distance`, `intra_class_dispersion`, `centroid_separation` | Distanza euclidea tra i centroidi delle due classi, normalizzata per la dispersione intra-classe (effect size, simile a un Cohen's d multivariato) |

In tutte le funzioni le feature vengono standardizzate (`StandardScaler`,
media 0 / varianza 1) prima del calcolo, salvo diversa indicazione.

---

## 3. Cosa fa ciascuno script (pipeline)

### 3.1 `calculate_clusters_embedding_labels.py`

Per ogni `(model_conf, dataset)`:
1. Carica il CSV e calcola statistiche descrittive (n. campioni, n. feature,
   bilanciamento delle classi `CD=0`/`CD=1`).
2. Calcola le 4 famiglie di metriche sopra (qualità cluster, KNN agreement,
   recovery non supervisionato, separazione dei centroidi) usando `CD` come
   etichetta binaria.
3. Salva i risultati incrementali in `clustering_metrics_partial.csv` dopo
   ogni dataset (in caso di crash a metà esecuzione).

Al termine di tutte le combinazioni:
4. Salva tutti i risultati dettagliati.
5. Calcola e salva statistiche di riepilogo (media/deviazione standard)
   aggregate per configurazione modello e per dataset.
6. Genera tabelle pivot (`model_configuration` × `dataset`) per le metriche
   chiave, per facilitare il confronto tra pipeline diverse.

### 3.2 `calculate_clusters_embedding_dataset.py`

Per ogni `model_conf` (scoperta automaticamente):
1. Carica **tutti i dataset** e li concatena in un unico blocco di feature,
   assegnando a ciascun campione un indice di dataset (`dataset_labels`) e
   mantenendo la label originale (`cd_labels`, colonna `label_column`).
2. **[1/4] Separazione complessiva tra dataset**: tratta ogni dataset come un
   cluster e calcola qualità del clustering + KNN agreement sull'intero
   insieme (N-way, non binario).
3. **[2/4] Confronti a coppie tra dataset**: per ogni coppia di dataset,
   ricalcola tutte le metriche (incluse quelle "binarie" come recovery non
   supervisionato e separazione dei centroidi, valide solo per 2 classi).
4. **[3/4] Combinazioni dataset+label**: crea `2 × n_dataset` "cluster"
   (es. `chapman_ningbo_CD0`, `chapman_ningbo_CD1`, `cod15_CD0`, ...) e
   calcola qualità clustering + KNN agreement su questa segmentazione più
   fine.
5. **[4/4] Confronti a coppie tra combinazioni dataset+label**: per ogni
   coppia di combinazioni (es. `chapman_ningbo_CD0` vs `cod15_CD1`), calcola
   tutte le metriche binarie, annotando anche se la coppia condivide lo
   stesso dataset e/o la stessa label.

Al termine:
6. Costruisce, per ogni configurazione modello e per alcune metriche chiave
   (`knn_agreement`, `centroid_distance`, `centroid_separation`,
   `spectral_ari`, `gmm_ari`), una **matrice quadrata "a confusion matrix"**
   (combinazione × combinazione) con i valori pairwise, utile per
   visualizzare in un colpo d'occhio quali gruppi sono più/meno separati tra
   loro.

---

## 4. Output prodotto

Entrambi gli script:
- Reindirizzano `stdout` su un `Logger` che scrive **sia a terminale sia su
  file** (classe `Logger`), producendo un log testuale completo dell'esecuzione
  con timestamp nel nome (`*_log_YYYYMMDD_HHMMSS.txt`).
- Stampano progressivamente lo stato di avanzamento (dataset caricati,
  metriche calcolate, eventuali errori catturati per singola metrica senza
  interrompere l'esecuzione — in caso di errore la metrica viene salvata come
  `NaN`).

### 4.1 Output di `calculate_clusters_embedding_labels.py`

Nella cartella `output_dir` (`/home/edofroses/ecg_fm/clustering_metrics_results_labels`):

| File | Contenuto |
|---|---|
| `clustering_evaluation_log_<timestamp>.txt` | Log completo dell'esecuzione |
| `clustering_metrics_partial.csv` | Risultati salvati incrementalmente (checkpoint) |
| `clustering_metrics_detailed.csv` | Una riga per ogni `(model_configuration, dataset)`, con tutte le metriche calcolate |
| `clustering_metrics_summary_by_model.csv` | Media e deviazione standard di ogni metrica, aggregate per `model_configuration` |
| `clustering_metrics_summary_by_dataset.csv` | Media e deviazione standard di ogni metrica, aggregate per `dataset` |
| `pivot_<metric>.csv` (uno per `silhouette_score`, `knn_label_agreement_k10`, `spectral_ari`, `centroid_separation`) | Tabella pivot `model_configuration` × `dataset` con il valore della metrica |

Colonne principali di `clustering_metrics_detailed.csv`:
`model_configuration, dataset, n_samples, n_features, n_class_0, n_class_1,
class_balance_ratio, silhouette_score, davies_bouldin_score,
calinski_harabasz_score, knn_label_agreement_k10, spectral_ari, spectral_ami,
gmm_ari, gmm_ami, centroid_distance, intra_class_dispersion,
centroid_separation`.

### 4.2 Output di `calculate_clusters_embedding_dataset.py`

Nella cartella `output_dir` (`/home/edofroses/ecg_fm/clustering_metrics_results_datasets_<label_column>`):

| File | Contenuto |
|---|---|
| `clustering_evaluation_datasets_log_<timestamp>.txt` | Log completo dell'esecuzione |
| `confusion_matrix_style/<model_conf>_<metric>_matrix.csv` (per ogni configurazione modello e ogni metrica chiave) | Matrice quadrata combinazione×combinazione con i valori pairwise della metrica |

Lo script tiene traccia internamente anche di risultati più granulari
(separazione complessiva tra dataset, confronti a coppie tra dataset,
combinazioni dataset+label) tramite le liste `all_results`,
`pairwise_results`, `dataset_label_results`, ma **solo le matrici
combinazione×combinazione derivate dai confronti dataset+label vengono
effettivamente salvate su disco** in questa versione dello script.

---

## 5. Requisiti

```
pandas
numpy
scipy
scikit-learn
```

## 6. Esecuzione

Prima di eseguire, modificare nel blocco `if __name__ == '__main__':` di
ciascuno script i valori hard-coded rilevanti per il proprio ambiente:
`base_path`, `output_dir`, `datasets`, `models_conf` (o `label_column`),
e il nome della colonna etichetta se diverso da `CD`/il valore di
`label_column`.

```bash
python calculate_clusters_embedding_labels.py
python calculate_clusters_embedding_dataset.py
```
