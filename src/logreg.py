"""
src/logreg.py — Multinomial Logistic Regression implemented from scratch in NumPy.

Model:    z = XW + b,  p = softmax(z)
Loss:     L = -(1/N) sum_i log p[i, y_i] + lambda * ||W||^2
Gradient: dL/dW = X.T @ (P - Y_onehot) / N + 2*lambda*W
          dL/db = (P - Y_onehot).mean(axis=0)
Optimiser: mini-batch SGD with momentum (beta=0.9)
"""
import numpy as np


class SoftmaxRegression:

    def __init__(self, n_features, n_classes, lr=0.01, lam=1e-4, momentum=0.9, seed=42):
        self.lr = lr
        self.lam = lam
        self.momentum = momentum
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_features, n_classes)) * 0.01
        self.b = np.zeros(n_classes)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    # ------------------------------------------------------------------ #
    # Core maths
    # ------------------------------------------------------------------ #

    def _softmax(self, logits):
        logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        return self._softmax(X @ self.W + self.b)

    def loss(self, X, y):
        N = len(y)
        P = self.forward(X)
        log_p = np.log(P[np.arange(N), y] + 1e-12)
        ce  = -log_p.mean()
        reg = self.lam * np.sum(self.W ** 2)
        return ce + reg

    def _gradient(self, X, y):
        N = len(y)
        P = self.forward(X)
        Y_oh = np.zeros_like(P)
        Y_oh[np.arange(N), y] = 1.0
        delta = (P - Y_oh) / N
        grad_W = X.T @ delta + 2 * self.lam * self.W
        grad_b = delta.sum(axis=0)
        return grad_W, grad_b

    def _step(self, grad_W, grad_b):
        self.vW = self.momentum * self.vW - self.lr * grad_W
        self.vb = self.momentum * self.vb - self.lr * grad_b
        self.W += self.vW
        self.b += self.vb

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def fit(self, X_train, y_train, X_val, y_val,
            batch_size=128, max_epochs=100, patience=10, verbose=True):
        N = len(y_train)
        best_val_loss = np.inf
        best_W = self.W.copy()
        best_b = self.b.copy()
        wait = 0

        history = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  []}

        for epoch in range(max_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                batch = idx[start:start + batch_size]
                gW, gb = self._gradient(X_train[batch], y_train[batch])
                self._step(gW, gb)

            tl = self.loss(X_train, y_train)
            vl = self.loss(X_val, y_val)
            ta = self.accuracy(X_train, y_train)
            va = self.accuracy(X_val, y_val)

            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(ta)
            history["val_acc"].append(va)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"train_loss={tl:.4f}  val_loss={vl:.4f} | "
                      f"train_acc={ta:.4f}  val_acc={va:.4f}")

            if vl < best_val_loss:
                best_val_loss = vl
                best_W = self.W.copy()
                best_b = self.b.copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no improvement for {patience} epochs)")
                    break

        self.W = best_W
        self.b = best_b
        return history

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path):
        np.savez(path, W=self.W, b=self.b)
        print(f"Model saved → {path}.npz")

    @classmethod
    def load(cls, path, n_features, n_classes):
        data = np.load(path)
        m = cls(n_features, n_classes)
        m.W = data["W"]
        m.b = data["b"]
        return m
