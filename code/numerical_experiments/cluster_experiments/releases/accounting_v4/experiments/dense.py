"""Streamed or memory-resident dense NPY designs without a Gram matrix."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import expit


class DenseDesign:
    def __init__(
        self,
        path,
        block_rows=128,
        transpose=False,
        row_scale=None,
        column_scale=None,
        resident=False,
        max_resident_bytes=32 * 1024**3,
        _resident=None,
    ):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError("streamed NPY requires format version 1 or 2")
            self.offset = f.tell()
        if len(shape) != 2 or min(shape) < 1 or fortran:
            raise ValueError("streamed NPY must be a nonempty C-order 2D array")
        if dtype.kind != "f" or dtype.itemsize not in (4, 8):
            raise ValueError("streamed NPY must be real float32 or float64")
        if self.path.stat().st_size != self.offset + np.prod(shape) * dtype.itemsize:
            raise ValueError("truncated or unexpected streamed NPY payload")
        if not isinstance(block_rows, int) or block_rows < 1:
            raise ValueError("block_rows must be positive")
        self.base_shape, self.dtype, self.block_rows = shape, dtype, block_rows
        self.transposed = transpose
        self.row_scale = np.ones(shape[0]) if row_scale is None else row_scale
        self.column_scale = np.ones(shape[1]) if column_scale is None else column_scale
        self.shape = shape[::-1] if transpose else shape
        self._frobenius = None
        self._resident = _resident
        if resident and self._resident is None:
            required = int(np.prod(shape)) * 8
            if required > max_resident_bytes:
                raise MemoryError("resident float64 design exceeds its payload budget")
            # Convert bounded row blocks once. Subsequent BLAS products avoid
            # both file reads and hidden full-matrix float32-to-float64 copies.
            array = np.empty(shape, dtype=np.float64)
            for start, block in self.blocks():
                array[start : start + len(block)] = block
            array.flags.writeable = False
            self._resident = array

    def blocks(self):
        rows, cols = self.base_shape
        if self._resident is not None:
            for start in range(0, rows, self.block_rows):
                yield start, self._resident[start : start + self.block_rows]
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            for start in range(0, rows, self.block_rows):
                count = min(self.block_rows, rows - start)
                block = np.fromfile(f, dtype=self.dtype, count=count * cols)
                if block.size != count * cols:
                    raise ValueError("truncated streamed matrix")
                yield start, block.reshape(count, cols)

    @property
    def T(self):
        out = DenseDesign(
            self.path,
            self.block_rows,
            not self.transposed,
            self.row_scale,
            self.column_scale,
            _resident=self._resident,
        )
        out._frobenius = self._frobenius
        return out

    def __matmul__(self, value):
        value = np.asarray(value)
        if value.ndim not in (1, 2) or value.shape[0] != self.shape[1]:
            raise ValueError("incompatible streamed matrix product")
        shape = (self.shape[0],) + value.shape[1:]
        if self._resident is not None:
            before = self.row_scale if self.transposed else self.column_scale
            after = self.column_scale if self.transposed else self.row_scale
            matrix = self._resident.T if self.transposed else self._resident
            scaled = before * value if value.ndim == 1 else before[:, None] * value
            product = matrix @ scaled
            return after * product if value.ndim == 1 else after[:, None] * product
        out = np.zeros(shape, dtype=np.result_type(value.dtype, np.float64))
        if not self.transposed:
            scaled = (
                self.column_scale * value
                if value.ndim == 1
                else (self.column_scale[:, None] * value)
            )
            for start, block in self.blocks():
                product = block @ scaled
                # Keep all temporary arrays bounded by the row block size.
                scale = self.row_scale[start : start + len(block)]
                out[start : start + len(block)] = (
                    scale * product if value.ndim == 1 else scale[:, None] * product
                )
        else:
            for start, block in self.blocks():
                scale = self.row_scale[start : start + len(block)]
                piece = value[start : start + len(block)]
                piece = scale * piece if value.ndim == 1 else scale[:, None] * piece
                out += block.T @ piece
            out *= self.column_scale if value.ndim == 1 else self.column_scale[:, None]
        return out

    def validate(self):
        frobenius, nnz = 0.0, 0
        for start, block in self.blocks():
            if not np.all(np.isfinite(block)):
                raise ValueError("nonfinite dense design")
            nnz += np.count_nonzero(block)
            scaled = (
                block
                * self.column_scale
                * self.row_scale[start : start + len(block), None]
            )
            frobenius += np.einsum("ij,ij->", scaled, scaled)
        if frobenius <= 0 or not np.isfinite(frobenius):
            raise ValueError("dense design must be finite and nonzero")
        self._frobenius, self.nnz = float(frobenius), int(nnz)
        return self

    def gram_product(self, value):
        """Apply A.T A with one pass over native design blocks, never form it."""
        if self.transposed:
            return self.T @ (self @ value)
        value = np.asarray(value)
        two = value.ndim == 2
        scaled = (
            self.column_scale[:, None] * value if two else self.column_scale * value
        )
        result = np.zeros_like(value, dtype=float)
        for start, block in self.blocks():
            block = np.asarray(block, dtype=float)
            product = block @ scaled
            scale = self.row_scale[start : start + len(block)] ** 2
            product *= scale[:, None] if two else scale
            result += block.T @ product
        return result * (self.column_scale[:, None] if two else self.column_scale)

    def loss_gradient(self, coef, target, logistic=False):
        """Fuse prediction, loss, and adjoint accumulation over the same block."""
        if self.transposed:
            prediction = self @ coef
            if logistic:
                margins = target * prediction
                return (
                    float(np.logaddexp(0, -margins).mean()),
                    self.T @ (-target * expit(-margins) / len(target)),
                )
            residual = prediction - target
            return float(0.5 * residual @ residual), self.T @ residual
        scaled = self.column_scale * coef
        result = np.zeros(self.shape[1])
        loss = 0.0
        for start, block in self.blocks():
            block = np.asarray(block, dtype=float)
            scale = self.row_scale[start : start + len(block)]
            prediction = (block @ scaled) * scale
            labels = target[start : start + len(block)]
            if logistic:
                margins = labels * prediction
                loss += float(np.logaddexp(0, -margins).sum() / len(target))
                weights = -labels * expit(-margins) / len(target)
            else:
                weights = prediction - labels
                loss += float(0.5 * weights @ weights)
            result += block.T @ (scale * weights)
        return loss, result * self.column_scale

    def normalized(self):
        sums = np.zeros(self.shape[1])
        for start, block in self.blocks():
            scaled = (
                block
                * self.column_scale
                * self.row_scale[start : start + len(block), None]
            )
            if self.transposed:
                sums[start : start + len(block)] = np.einsum("ij,ij->i", scaled, scaled)
            else:
                sums += np.einsum("ij,ij->j", scaled, scaled)
        factors = 1 / np.maximum(np.sqrt(sums), 1e-15)
        out = DenseDesign(
            self.path,
            self.block_rows,
            self.transposed,
            self.row_scale * factors if self.transposed else self.row_scale,
            self.column_scale if self.transposed else self.column_scale * factors,
            _resident=self._resident,
        )
        out._frobenius = float(np.count_nonzero(sums))
        return out

    def norm_bound(self):
        if self._frobenius is None:
            self.validate()
        return self._frobenius  # certified ||A||_2^2 <= ||A||_F^2

    def to_sparse(self, max_bytes=2 * 1024**3):
        # Dense-valued CSR plus input/assembly workspaces: conservative guard.
        if self.shape[0] * self.shape[1] * 24 > max_bytes:
            raise MemoryError(
                "explicit dense design exceeds baseline representation budget"
            )
        blocks = []
        for start, block in self.blocks():
            scaled = (
                block
                * self.column_scale
                * self.row_scale[start : start + len(block), None]
            )
            blocks.append(sp.csr_matrix(scaled))
        out = sp.vstack(blocks, format="csr")
        return out.T.tocsr() if self.transposed else out
