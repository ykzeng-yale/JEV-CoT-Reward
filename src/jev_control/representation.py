"""Training-only text/numeric representation reusable in prospective control."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FrozenRepresentation:
    vectorizer: TfidfVectorizer | None
    imputer: SimpleImputer
    scaler: StandardScaler
    numeric_width: int
    vocabulary_size: int

    def transform(self, documents, numeric):
        """Transform only: unseen deployment examples never refit statistics."""
        numeric = np.asarray(numeric, dtype=float)
        if numeric.ndim != 2 or numeric.shape != (len(documents), self.numeric_width):
            raise ValueError("Document count or frozen numeric schema width mismatch")
        if np.isinf(numeric).any():
            raise ValueError("Infinite numeric features are not supported")
        text = (self.vectorizer.transform(documents) if self.vectorizer is not None
                else sparse.csr_matrix((len(documents), 0)))
        numbers = self.scaler.transform(self.imputer.transform(numeric))
        return sparse.hstack([text, sparse.csr_matrix(numbers)], format="csr")


def fit_representation(documents, numeric):
    """Fit once on training problems; return the reusable transform and matrix."""
    numeric = np.asarray(numeric, dtype=float)
    if numeric.ndim != 2 or numeric.shape[0] != len(documents) or not len(documents):
        raise ValueError("Nonempty aligned training documents and numeric rows required")
    if np.isinf(numeric).any():
        raise ValueError("Infinite numeric features are not supported")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4096, sublinear_tf=True)
    try:
        text = vectorizer.fit_transform(documents)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        vectorizer = None
        text = sparse.csr_matrix((len(documents), 0))
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    numbers = scaler.fit_transform(imputer.fit_transform(numeric))
    fitted = FrozenRepresentation(vectorizer, imputer, scaler, numeric.shape[1], text.shape[1])
    return fitted, sparse.hstack([text, sparse.csr_matrix(numbers)], format="csr")
