"""Matrix-free problems over sparse or row-streamed dense designs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from .data import digest
from .dense import DenseDesign
from .matrices import load_matrix


def norm_bound(a):
    """Certified ||A||_2^2 upper bound, not a power-iteration underestimate."""
    if isinstance(a, DenseDesign):
        return a.norm_bound()
    return float(abs(a).sum(axis=0).max() * abs(a).sum(axis=1).max())


@dataclass
class Problem:
    family: str
    n: int
    L: float
    mu: float
    q: np.ndarray | None = None
    Q: object = None
    A: object = None
    target: object = None
    indices: object = None
    truth: object = None
    regularization: float = 0.0
    target_scale: float | None = None
    provenance: str = "synthetic"

    def hess(self, x):
        if self.family == "qp":
            return self.Q @ x
        if self.family == "lasso":
            d = self.A.shape[1]
            v = (
                self.A.gram_product(x[:d] - x[d:])
                if isinstance(self.A, DenseDesign)
                else self.A.T @ (self.A @ (x[:d] - x[d:]))
            )
            return np.concatenate((v, -v))
        raise ValueError("logistic Hessian is not constant")

    def gradient(self, x):
        if self.family in ("lasso", "logistic") and isinstance(self.A, DenseDesign):
            return self.gradient_and_value(x)[0]
        if self.family != "logistic":
            return self.hess(x) + self.q
        margins = self.target * (self.A @ x)
        return (
            -(self.A.T @ (self.target * expit(-margins))) / self.A.shape[0]
            + self.regularization * x
        )

    def gradient_and_value(self, x):
        if self.family in ("lasso", "logistic") and isinstance(self.A, DenseDesign):
            coef = (
                x[: self.A.shape[1]] - x[self.A.shape[1] :]
                if self.family == "lasso"
                else x
            )
            value, gradient = self.A.loss_gradient(
                coef, self.target, logistic=self.family == "logistic"
            )
            if self.family == "lasso":
                return (
                    np.r_[
                        gradient + self.regularization, -gradient + self.regularization
                    ],
                    value + self.regularization * float(np.sum(x)),
                )
            return (
                gradient + self.regularization * x,
                value + 0.5 * self.regularization * float(x @ x),
            )
        return self.gradient(x), None

    def project(self, x):
        z = x.copy()
        if self.family == "qp":
            z[self.indices] = self.target
        elif self.family == "lasso":
            np.maximum(z, 0, out=z)
        return z

    def objective(self, x):
        if self.family == "qp":
            return float(0.5 * x @ self.hess(x) + self.q @ x)
        if self.family == "lasso":
            d = self.A.shape[1]
            r = self.A @ (x[:d] - x[d:]) - self.target
            return float(0.5 * r @ r + self.regularization * np.sum(x))
        return float(
            np.logaddexp(0, -self.target * (self.A @ x)).mean()
            + 0.5 * self.regularization * (x @ x)
        )

    def metrics(self, x, gradient=None, objective=None):
        """Same normalized original-problem stationarity test for every solver."""
        if gradient is None and isinstance(self.A, DenseDesign):
            gradient, objective = self.gradient_and_value(x)
        if self.family == "lasso":
            d = self.A.shape[1]
            coef = x[:d] - x[d:]
            r = (
                self.A @ coef - self.target
                if objective is None or gradient is None
                else None
            )
            g = self.A.T @ r if gradient is None else gradient[:d] - self.regularization
            ell = self.L / 2
            v = coef - g / ell
            prox = np.sign(v) * np.maximum(abs(v) - self.regularization / ell, 0)
            residual = ell * np.linalg.norm(coef - prox)
            scale = self.target_scale
            if scale is None:
                scale = 1 + np.linalg.norm(self.A.T @ self.target)
            feasibility = float(np.linalg.norm(np.minimum(x, 0)))
            if objective is None:
                objective = float(0.5 * r @ r + self.regularization * np.sum(x))
        else:
            if self.family == "logistic":
                margins = (
                    self.target * (self.A @ x)
                    if objective is None or gradient is None
                    else None
                )
                g = (
                    gradient
                    if gradient is not None
                    else (
                        -(self.A.T @ (self.target * expit(-margins))) / self.A.shape[0]
                        + self.regularization * x
                    )
                )
                if objective is None:
                    objective = float(
                        np.logaddexp(0, -margins).mean()
                        + 0.5 * self.regularization * (x @ x)
                    )
            else:
                g = self.gradient(x) if gradient is None else gradient
                objective = float(0.5 * x @ (g - self.q) + self.q @ x)
            residual = self.L * np.linalg.norm(x - self.project(x - g / self.L))
            scale = 1 + np.linalg.norm(self.q) if self.q is not None else 1
            feasibility = (
                float(np.linalg.norm(x[self.indices] - self.target))
                / (1 + np.linalg.norm(self.target))
                if self.family == "qp"
                else 0.0
            )
        return dict(
            objective=objective,
            stationarity=float(residual / scale),
            feasibility=feasibility,
            solution_error=(
                float(np.linalg.norm(x - self.truth) / (1 + np.linalg.norm(self.truth)))
                if self.truth is not None
                else float("nan")
            ),
        )


def make_qp(seed=0, side=40, kappa=1e4):
    """Periodic 3D Laplacian: exact spectral endpoints for even side >= 4.

    Q = I/kappa + (1-1/kappa) Laplacian/12. The all-ones and
    checkerboard modes certify lambda_min=1/kappa and lambda_max=1.
    Random coordinate equalities have an exact, factor-free projector.
    """
    if side < 4 or side % 2 or kappa < 1e4:
        raise ValueError("QP needs even side >= 4 and kappa >= 1e4")
    rng = np.random.default_rng(seed)
    t = sp.diags(
        [-np.ones(side - 1), 2 * np.ones(side), -np.ones(side - 1)],
        [-1, 0, 1],
        format="lil",
    )
    t[0, -1] = t[-1, 0] = -1
    eye = sp.eye(side, format="csr")
    lap = (
        sp.kron(sp.kron(t, eye), eye)
        + sp.kron(sp.kron(eye, t), eye)
        + sp.kron(sp.kron(eye, eye), t)
    ).tocsr()
    n = side**3
    Q = (sp.eye(n, format="csr") / kappa + (1 - 1 / kappa) * lap / 12).tocsr()
    indices = np.sort(rng.choice(n, n // 4, replace=False))
    truth = rng.normal(size=n) / np.sqrt(n)
    dual = rng.normal(size=len(indices)) / np.sqrt(len(indices))
    q = -(Q @ truth)
    q[indices] -= dual
    return Problem(
        "qp",
        n,
        1.0,
        1 / kappa,
        q=q,
        Q=Q,
        indices=indices,
        target=truth[indices],
        truth=truth,
    )


def make_regression(
    family,
    matrix,
    seed,
    orientation="native",
    lambda_ratio=0.01,
    targets=None,
    normalization="column_l2",
    positive_class=None,
    lambda_value=None,
):
    rng = np.random.default_rng(seed)
    A = (
        matrix
        if isinstance(matrix, DenseDesign)
        else sp.csr_matrix(matrix, dtype=float)
    )
    if orientation == "transpose":
        A = A.T if isinstance(A, DenseDesign) else A.T.tocsr()
    # Column normalization preserves sparsity and is recorded in provenance.
    if normalization == "column_l2":
        if isinstance(A, DenseDesign):
            A = A.normalized()
        else:
            norms = np.sqrt(np.asarray(A.power(2).sum(axis=0)).ravel())
            A = (A @ sp.diags(1 / np.maximum(norms, 1e-15))).tocsr()
    elif normalization != "none":
        raise ValueError("normalization must be column_l2 or none")
    ell = norm_bound(A)
    if ell <= 0 or not np.isfinite(ell):
        raise ValueError("empty/nonfinite design")
    m, d = A.shape
    if targets is not None:
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (m,) or not np.all(np.isfinite(targets)):
            raise ValueError("labels must be finite and match matrix rows")
        if orientation != "native":
            raise ValueError("native labels cannot be attached to a transposed design")
    coef = np.zeros(d)
    support = rng.choice(d, max(1, d // 100), replace=False)
    coef[support] = rng.normal(size=len(support))
    if family == "lasso":
        target = A @ coef + 0.01 * rng.normal(size=m) if targets is None else targets
        atb = A.T @ target
        lam = (
            lambda_ratio * float(abs(atb).max())
            if lambda_value is None
            else float(lambda_value)
        )
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("LASSO lambda must be positive")
        return Problem(
            family,
            2 * d,
            2 * ell,
            0.0,
            q=np.concatenate((-atb + lam, atb + lam)),
            A=A,
            target=target,
            regularization=lam,
            target_scale=float(1 + np.linalg.norm(atb)),
        )
    # SuiteSparse supplies designs, not classification labels. Seeded synthetic
    # Bernoulli labels are explicit; ridge ensures a finite minimizer.
    if targets is None:
        scores = A @ coef
        scores /= max(float(np.std(scores)), 1e-12)
        target = np.where(rng.random(m) < expit(scores), 1.0, -1.0)
    elif positive_class is not None:
        target = np.where(targets == positive_class, 1.0, -1.0)
    elif set(np.unique(targets)) == {-1.0, 1.0}:
        target = targets
    elif set(np.unique(targets)) == {0.0, 1.0}:
        target = 2 * targets - 1
    else:
        raise ValueError("multiclass labels require an explicit positive_class")
    if targets is not None and len(np.unique(target)) != 2:
        raise ValueError("logistic labels must include both classes")
    ridge = ell / (4 * m * 1e4)
    return Problem(
        family,
        d,
        ell / (4 * m) + ridge,
        ridge,
        A=A,
        target=target,
        regularization=ridge,
    )


def build(config, data_dir):
    if config["family"] == "qp":
        return make_qp(config["seed"], config["side"], config.get("kappa", 1e4))
    matrix, metadata = load_matrix(config, data_dir)
    spec = config.get("matrix_source", {})
    targets = None
    response = spec.get("response", "synthetic")
    if response == "native":
        path = Path(spec["labels_path"])
        path = path if path.is_absolute() else Path(data_dir) / path
        if "labels_sha256" in spec and digest(path) != spec["labels_sha256"]:
            raise ValueError("label checksum mismatch")
        targets = np.load(path, allow_pickle=False)
    elif response != "synthetic":
        raise ValueError("response must be native or synthetic")
    p = make_regression(
        config["family"],
        matrix,
        config["seed"],
        config["orientation"],
        config.get("lambda_ratio", 0.01),
        targets=targets,
        normalization=spec.get("normalization", "column_l2"),
        positive_class=spec.get("positive_class"),
        lambda_value=spec.get("lasso_lambda"),
    )
    p.provenance = json.dumps(
        dict(
            dataset=config["dataset"],
            matrix=metadata,
            orientation=config["orientation"],
            normalization=spec.get("normalization", "column_l2"),
            response=response,
            positive_class=spec.get("positive_class"),
            response_seed=config["seed"] if response == "synthetic" else None,
        ),
        sort_keys=True,
    )
    return p
