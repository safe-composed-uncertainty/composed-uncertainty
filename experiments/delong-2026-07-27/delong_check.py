"""External validation of the paired-bootstrap cross-covariance against DeLong.

Item A10 of the upgrade plan. In the binary case the Rank Graduation Accuracy
of a model coincides with the area under its ROC curve, and the German-credit
chain is two models scoring the same 300 held-out rows -- exactly the setting
DeLong, DeLong and Clarke-Pearson (1988) derived their covariance estimator
for. Their estimator is analytic and shares no code, no resampling and no
random draw with our paired bootstrap, so agreement between the two is an
external check on the machinery this article rests on.

Reads only the persisted replicates of the pinned redraw driver; writes its
own results file and prints every quantity it compares.
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPL = HERE.parent / 'redraw-2026-07-24' / 'replicates-redraw.npz'
OUT = HERE / 'results-delong.json'


def auc(y, s):
    """Area under the ROC curve by the Mann-Whitney statistic, ties at 0.5."""
    pos, neg = s[y == 1], s[y == 0]
    diff = pos[:, None] - neg[None, :]
    return (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos.size * neg.size)


def midrank(x):
    """Ranks with ties averaged, as DeLong's structural components require."""
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    n = x.size
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


def delong_cov(y, scores):
    """DeLong covariance of several AUCs on one shared sample.

    scores is (k, n). Returns the k-by-k covariance of the AUC estimates,
    following the structural-component formulation of DeLong et al. (1988) in
    the fast midrank form of Sun and Xu (2014).
    """
    pos_mask = y == 1
    m, n = int(pos_mask.sum()), int((~pos_mask).sum())
    k = scores.shape[0]

    v01 = np.empty((k, m))     # components over the positive cases
    v10 = np.empty((k, n))     # components over the negative cases
    aucs = np.empty(k)

    for r in range(k):
        pos, neg = scores[r][pos_mask], scores[r][~pos_mask]
        tx, ty = midrank(pos), midrank(neg)
        tz = midrank(np.concatenate([pos, neg]))
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m

    s01 = np.cov(v01, ddof=1) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10, ddof=1) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    return aucs, np.atleast_2d(s01) / m + np.atleast_2d(s10) / n


def main():
    d = np.load(REPL)
    y = d['y_test'].astype(int)
    p_logit, p_rf = d['p_full_logit'], d['p_full_rf']

    # RGA = AUROC on binary outcomes: check the identity before using it.
    rga_point = {'logit': float(d['point_logit'][0]), 'rf': float(d['point_rf'][0])}
    auc_point = {'logit': auc(y, p_logit), 'rf': auc(y, p_rf)}
    identity_gap = {k: abs(rga_point[k] - auc_point[k]) for k in rga_point}

    # Ours: the paired bootstrap covariance of the two models' RGA on the
    # shared row stream. Column 0 of each chain vector is RGA.
    boot = np.column_stack([d['chain_vec_redraw_logit'][:, 0],
                            d['chain_vec_redraw_rf'][:, 0]])
    boot_cov = np.cov(boot, rowvar=False, ddof=1)
    boot_corr = boot_cov[0, 1] / np.sqrt(boot_cov[0, 0] * boot_cov[1, 1])

    # Theirs: DeLong, analytic, on the same 300 rows.
    aucs, dl_cov = delong_cov(y, np.vstack([p_logit, p_rf]))
    dl_corr = dl_cov[0, 1] / np.sqrt(dl_cov[0, 0] * dl_cov[1, 1])

    res = {
        'study': 'DeLong cross-check of the paired-bootstrap RGA covariance',
        'source_replicates': str(REPL.name),
        'n_test': int(y.size),
        'n_positive': int((y == 1).sum()),
        'B': int(boot.shape[0]),
        'rga_equals_auroc': {
            'rga_point': rga_point,
            'auroc_point': {k: float(v) for k, v in auc_point.items()},
            'max_abs_gap': float(max(identity_gap.values())),
        },
        'paired_bootstrap': {
            'cov': boot_cov.tolist(),
            'sd': [float(np.sqrt(boot_cov[0, 0])), float(np.sqrt(boot_cov[1, 1]))],
            'correlation': float(boot_corr),
        },
        'delong': {
            'auc': [float(a) for a in aucs],
            'cov': dl_cov.tolist(),
            'sd': [float(np.sqrt(dl_cov[0, 0])), float(np.sqrt(dl_cov[1, 1]))],
            'correlation': float(dl_corr),
        },
        'agreement': {
            'sd_ratio_bootstrap_over_delong': [
                float(np.sqrt(boot_cov[0, 0] / dl_cov[0, 0])),
                float(np.sqrt(boot_cov[1, 1] / dl_cov[1, 1])),
            ],
            'covariance_ratio': float(boot_cov[0, 1] / dl_cov[0, 1]),
            'correlation_difference': float(boot_corr - dl_corr),
        },
    }
    OUT.write_text(json.dumps(res, indent=1))

    print('RGA = AUROC identity, max |gap| = %.2e' % res['rga_equals_auroc']['max_abs_gap'])
    print('  logit RGA %.6f  AUROC %.6f' % (rga_point['logit'], auc_point['logit']))
    print('  rf    RGA %.6f  AUROC %.6f' % (rga_point['rf'], auc_point['rf']))
    print()
    print('paired bootstrap  sd %.6f / %.6f  corr %+.4f'
          % (*res['paired_bootstrap']['sd'], boot_corr))
    print('DeLong analytic   sd %.6f / %.6f  corr %+.4f'
          % (*res['delong']['sd'], dl_corr))
    print()
    print('sd ratio bootstrap/DeLong: %.4f, %.4f'
          % tuple(res['agreement']['sd_ratio_bootstrap_over_delong']))
    print('covariance ratio: %.4f' % res['agreement']['covariance_ratio'])
    print('correlation difference: %+.4f' % res['agreement']['correlation_difference'])
    print('wrote %s' % OUT)


if __name__ == '__main__':
    main()
