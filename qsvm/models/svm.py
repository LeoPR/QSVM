from sklearn.svm import SVC
import numpy as np

try:
    import joblib
except Exception:
    joblib = None


class ClassicalSVM:
    """
    SVM clássico (scikit-learn) encapsulado.
    Exemplo de uso:
      svm = ClassicalSVM(kernel="linear", C=1.0)
      svm.fit(X_train, y_train)
      y_pred = svm.predict(X_test)
    """
    def __init__(self, **kwargs):
        """
        Aceita os mesmos kwargs do sklearn.svm.SVC, por exemplo:
          - kernel: "linear", "rbf", "poly", "sigmoid"
          - C: float
          - gamma: "scale", "auto" ou float
          - degree: int (para poly)
          - probability: bool (se True, habilita predict_proba)
        """
        self.model = SVC(**kwargs)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if not getattr(self.model, "probability", False):
            raise RuntimeError("probability=False no SVC. Recrie com probability=True para usar predict_proba.")
        X = np.asarray(X)
        return self.model.predict_proba(X)

    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        return self.model.score(X, y)

    def save(self, path: str):
        if joblib is None:
            raise RuntimeError("joblib não disponível para salvar o modelo.")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        if joblib is None:
            raise RuntimeError("joblib não disponível para carregar o modelo.")
        model = joblib.load(path)
        obj = cls()
        obj.model = model
        return obj