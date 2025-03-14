import numpy as np
from tqdm import tqdm

class AgglomerativeHierarchicalClustering:
    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, initial_labels=None):
        """
        Wykonuje aglomeracyjną klasteryzację hierarchiczną.
        
        Parameters:
        -----------
        X: ndarray, shape (n_samples, n_features)
            Dane wejściowe.
        
        initial_labels: array-like, shape (n_samples,), optional
            Wstępne etykiety klastrów. Jeśli podano, algorytm zaczyna od tych klastrów.
        
        Returns:
        --------
        labels: ndarray, shape (n_samples,)
            Finalne etykiety po klasteryzacji.
        """
        n_samples = X.shape[0]
        
        if initial_labels is not None:
            self.clusters = self._initialize_clusters_from_labels(initial_labels)
        else:
            self.clusters = {i: [i] for i in range(n_samples)}

        unique_labels = np.unique(initial_labels)
        n_clusters_init = len(unique_labels)
            
        # Macierz odległości tylko dla unikalnych klastrów
        self.distances = np.full((n_clusters_init, n_clusters_init), np.inf)
        for i, label_i in enumerate(tqdm(unique_labels, desc="Comparing clusters", position=0)):
            for j in tqdm(range(i-1), desc=f"Calculating distances for cluster {label_i}/{len(unique_labels)}", position=1, leave=False):
                label_j = unique_labels[j]
                self.distances[i, j] = self._compute_cluster_distance(self.clusters[label_i], self.clusters[label_j], X)
                self.distances[j, i] = self.distances[i, j]

        with tqdm(total=len(self.clusters) - self.n_clusters, desc="Merging clusters") as pbar:
            while len(self.clusters) > self.n_clusters:
                min_dist = np.inf
                a, b = -1, -1
                clusters_keys = list(self.clusters.keys())
                
                for i in range(len(clusters_keys)): 
                    for j in range(i + 1, len(clusters_keys)):
                        dist = self.distances[clusters_keys[i], clusters_keys[j]]
                        if dist < min_dist:
                            min_dist = dist
                            a, b = clusters_keys[i], clusters_keys[j]

                if a in self.clusters and b in self.clusters:
                    self._merge_clusters(a, b)
                    self._update_distance_matrix(a, b, X)
                    pbar.update(1)
        
        return self._get_labels(n_samples)

    def _initialize_clusters_from_labels(self, labels):
        """Inicjalizuje klastry na podstawie wstępnych etykiet."""
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        return clusters

    def _compute_distance(self, x1, x2):
        """Oblicza odległość euklidesową między dwoma punktami."""
        return np.linalg.norm(x1 - x2)

    def _compute_cluster_distance(self, cluster_a, cluster_b, X):
        """Oblicza odległość między dwoma klastrami w zależności od wybranej metody łączenia."""
        points_a = X[cluster_a]
        points_b = X[cluster_b]
        
        if self.linkage == "single":
            return np.min([np.linalg.norm(pa - pb) for pa in points_a for pb in points_b])
        elif self.linkage == "complete":
            return np.max([np.linalg.norm(pa - pb) for pa in points_a for pb in points_b])
        elif self.linkage == "average":
            return np.mean([np.linalg.norm(pa - pb) for pa in points_a for pb in points_b])
        elif self.linkage == "ward":
            mean_a = np.mean(points_a, axis=0)
            mean_b = np.mean(points_b, axis=0)
            return np.linalg.norm(mean_a - mean_b)
        else:
            raise ValueError("Nieprawidłowa metoda łączenia klastrów.")

    def _merge_clusters(self, cluster_a, cluster_b):
        """Łączy dwa klastry w jeden."""
        self.clusters[cluster_a].extend(self.clusters.pop(cluster_b))

    def _update_distance_matrix(self, cluster_a, cluster_b, X):
        """
        Aktualizuje macierz odległości po połączeniu dwóch klastrów.
        
        Parameters:
        -----------
        cluster_a: int
            Indeks klastra, który pozostaje po połączeniu (połączenie z cluster_b).
        
        cluster_b: int
            Indeks klastra, który został połączony z cluster_a i zostanie usunięty.
        
        X: ndarray
            Dane wejściowe, aby móc ponownie obliczyć odległości między klastrami.
        """
        clusters_keys = list(self.clusters.keys())

        for i in clusters_keys:
            if i != cluster_a and i in self.clusters:
                new_dist = self._compute_cluster_distance(self.clusters[cluster_a], self.clusters[i], X)
                self.distances[cluster_a, i] = new_dist
                self.distances[i, cluster_a] = new_dist

        if cluster_b in clusters_keys:
            cluster_b_index = clusters_keys.index(cluster_b)

            self.distances = np.delete(self.distances, cluster_b_index, axis=0)
            self.distances = np.delete(self.distances, cluster_b_index, axis=1)

    def _get_labels(self, n_samples):
        """Zwraca finalne etykiety klastrów po zakończeniu procesu łączenia."""
        labels = np.zeros(n_samples, dtype=int)

        cluster_mapping = {cluster_id: idx + 1 for idx, cluster_id in enumerate(self.clusters.keys())}

        for cluster_id, points in self.clusters.items():
            for point in points:
                labels[point] = cluster_mapping[cluster_id]
        
        return labels
