"""Exercise the original rank kernel without importing its research runner."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def original_functions():
    path = ROOT / "research" / "run_bran_overnight_diagnostic_v1.py"
    tree = ast.parse(path.read_text())
    wanted = {"fold_weighted_auc", "_weighted_auc_draws"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise AssertionError("Original metric definitions are missing")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    scope = {"np": np}
    exec(compile(module, str(path), "exec"), scope)
    return scope


class OriginalMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = original_functions()

    def test_fold_auc_matches_independent_sklearn_calculation(self):
        rng = np.random.default_rng(20261001)
        folds = np.repeat(np.arange(5), 12)
        y = np.tile([0, 1], 30)
        p = np.round(rng.uniform(size=60), 1)
        observed = np.ones(60, bool)
        observed[[2, 17, 28]] = False
        expected, weights = [], []
        for fold in range(5):
            subset = observed & (folds == fold)
            expected.append(roc_auc_score(y[subset], p[subset]))
            weights.append(subset.sum())
        actual = self.functions["fold_weighted_auc"](y, p, observed, folds)
        self.assertAlmostEqual(actual, np.average(expected, weights=weights), places=13)

    def test_bootstrap_rank_kernel_matches_weighted_auc_and_exact_ties(self):
        rng = np.random.default_rng(107)
        folds = np.repeat(np.arange(5), 10)
        y = np.tile([0, 1], 25)
        p = np.round(rng.uniform(size=50), 1)
        observed = np.ones(50, bool)
        counts = np.ones((8, 50), dtype=np.int64)
        for draw in range(1, 8):
            for fold in range(5):
                idx = np.flatnonzero(folds == fold)
                counts[draw, idx] = np.bincount(rng.choice(10, 10, replace=True), minlength=10)
        actual = self.functions["_weighted_auc_draws"](y, p, observed, folds, counts)
        expected = []
        for count in counts:
            values, weights = [], []
            for fold in range(5):
                idx = folds == fold
                w = count[idx]
                if w[y[idx] == 0].sum() and w[y[idx] == 1].sum():
                    values.append(roc_auc_score(y[idx], p[idx], sample_weight=w))
                    weights.append(w.sum())
            expected.append(np.average(values, weights=weights))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
        tied = self.functions["_weighted_auc_draws"](y, np.full(50, .5), observed, folds, counts)
        np.testing.assert_allclose(tied, .5, rtol=0, atol=0)

    def test_shared_draws_preserve_zero_paired_contrast_for_identical_arms(self):
        folds = np.repeat(np.arange(5), 8)
        y = np.tile([0, 1], 20)
        p = np.linspace(.1, .9, 40)
        counts = np.ones((6, 40), np.int64)
        kernel = self.functions["_weighted_auc_draws"]
        first = kernel(y, p, np.ones(40, bool), folds, counts)
        second = kernel(y, p.copy(), np.ones(40, bool), folds, counts)
        np.testing.assert_array_equal(first - second, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
