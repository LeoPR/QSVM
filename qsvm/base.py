from abc import ABC, abstractmethod
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone
import numpy as np

class BaseModel(ABC):
    """
    Interface mínima para modelos no pacote qsvm.
    Implementações devem prover fit(X, y) e predict(X).
    """
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class MultiOutputWrapper:
    """
    Wrapper que permite treinar modelos multioutput de duas formas:
      - mode='multioutput': usa sklearn.multioutput.MultiOutputClassifier
      - mode='one-per-output': treina N modelos independentes (clone do estimator)
    Exemplo:
      wrapper = MultiOutputWrapper(estimator, mode='one-per-output')
      wrapper.fit(X, Y)  # Y shape (n_samples, n_outputs)
      Ypred = wrapper.predict(X)  # shape (n_samples, n_outputs)
    """
    def __init__(self, estimator, mode="multioutput"):
        assert mode in ("multioutput", "one-per-output")
        self.mode = mode
        self.base = estimator
        self._models = None
        self._multi = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            # single output, just fit base estimator
            self.base.fit(X, Y)
            self._models = None
            self._multi = None
            return self
        n_outputs = Y.shape[1]
        if self.mode == "multioutput":
            self._multi = MultiOutputClassifier(clone(self.base))
            self._multi.fit(X, Y)
            self._models = None
        else:
            # one-per-output
            self._models = []
            for j in range(n_outputs):
                m = clone(self.base)
                m.fit(X, Y[:, j])
                self._models.append(m)
            self._multi = None
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self._multi is not None:
            return self._multi.predict(X)
        if self._models is None:
            # base single-output model
            return self.base.predict(X)
        # stack predictions from each model
        preds = [m.predict(X) for m in self._models]
        return np.vstack(preds).T

    def predict_proba(self, X):
        X = np.asarray(X)
        if self._multi is not None:
            return self._multi.predict_proba(X)
        if self._models is None:
            if hasattr(self.base, "predict_proba"):
                return self.base.predict_proba(X)
            raise RuntimeError("No predict_proba available")
        # list of arrays each (n_samples, n_classes). Return list.
        return [m.predict_proba(X) for m in self._models]