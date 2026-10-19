from typing import Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> dict:
        """Reduce the dimensions of session.feature_matrix using PCA, t-SNE, or UMAP, store the coordinates, and plot a 2D scatter plot colored by category.

        Args:
            method: Dimensionality reduction technique ("pca", "tsne", or "umap").
            n_components: Number of dimensions to reduce to.
            random_state: Random seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.

        Returns:
            A dictionary containing the result ID, method, reduced coordinates, and explained variance ratio if applicable.
        """
        if session.feature_matrix is None:
            return {"error": "feature_matrix is None. Build DTM first."}
        
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        explained_variance = None
        if method == "pca":
            kwargs = {}
            if random_state is not None:
                kwargs["random_state"] = random_state
            reducer = PCA(n_components=n_components, **kwargs)
            coords = reducer.fit_transform(X_dense)
            explained_variance = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            kwargs = {}
            if random_state is not None:
                kwargs["random_state"] = random_state
            n_samples = X_dense.shape[0]
            curr_perplexity = perplexity if perplexity < n_samples else max(1.0, float(n_samples - 1))
            reducer = TSNE(n_components=n_components, perplexity=curr_perplexity, **kwargs)
            coords = reducer.fit_transform(X_dense)
        elif method == "umap":
            kwargs = {}
            if random_state is not None:
                kwargs["random_state"] = random_state
            n_samples = X_dense.shape[0]
            curr_neighbors = n_neighbors if n_neighbors < n_samples else max(2, n_samples - 1)
            reducer = umap.UMAP(n_components=n_components, n_neighbors=curr_neighbors, **kwargs)
            coords = reducer.fit_transform(X_dense)
        else:
            return {"error": f"Unknown method: {method}"}

        session.artifacts[f"reduced_{method}"] = coords

        # Plotting requirement: scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = session.dataframe["category_name"] if session.dataframe is not None and "category_name" in session.dataframe.columns else None
        
        if categories is not None:
            df_plot = pd.DataFrame({
                "dim1": coords[:, 0],
                "dim2": coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(coords[:, 0]),
                "category": categories
            })
            sns.scatterplot(data=df_plot, x="dim1", y="dim2", hue="category", palette="Set1", ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(coords[:, 0]))
        
        ax.set_title(f"{method.upper()} Dimensionality Reduction")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "coordinates": coords.tolist(),
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = explained_variance

        session.store_result("reduce_dimensions_tool", {"method": method, "n_components": n_components}, summary)
        return summary

    return [reduce_dimensions_tool]
