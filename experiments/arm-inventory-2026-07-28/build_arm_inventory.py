"""The arm inventory behind the numerical-agreement remark.

The article states that the width-inflation identity was checked against every
first-order quantity in the stored runs -- 46 arms -- that the bound of the
worst-case corollary holds on the 44 of them that carry a gradient, and that
the sign condition agrees on the 40 of those whose cross-term sum is non-zero.
Until now that claim had no artefact behind it: the counts lived only in the
prose. This script rebuilds the inventory from the deposited results files and
writes it out, so a referee can check the arithmetic rather than trust it.

For every arm it records where the arm comes from, whether it carries a
gradient, the share of variance the cross-terms contribute, and the two
first-order readings of that share -- the variance share V and the width
understatement U -- which Corollary 2 ties together by U = 1 - sqrt(1 - V).
The discrepancy column is the residual of that identity, in percentage points
of understatement.

Reads only deposited JSON; writes only into its own directory.
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
EXP = HERE.parent
OUT = HERE / 'arm-inventory.json'


def u_from_v(v_pct):
    """Corollary 2: the width understatement implied by a variance share."""
    v = v_pct / 100.0
    if v >= 1.0:
        return float('nan')
    return 100.0 * (1.0 - math.sqrt(1.0 - v))


def aggregation_arms():
    """Seven composites x two models x two conditioning schemes."""
    d = json.loads((EXP / 'aggregation-2026-07-25' / 'results-aggregation.json').read_text())
    rows = []
    for model, mv in d['models'].items():
        for arm, av in mv['arms'].items():
            for scheme, sv in av.items():
                if not isinstance(sv, dict):
                    continue
                v = sv.get('cross_covariance_share_of_variance_pct')
                if v is None:
                    continue
                rows.append({
                    'arm': f'aggregation/{model}/{arm}/{scheme}',
                    'source': 'aggregation-2026-07-25/results-aggregation.json',
                    'carries_gradient': True,
                    'gradient_is_one_hot': arm == 'shapley_min',
                    'cross_share_of_variance_pct': v,
                    'implied_understatement_pct': u_from_v(v),
                    'cross_term_sum_is_zero': abs(v) < 1e-12,
                })
    return rows


# The five composites of the volume study that the article reports a
# first-order understatement for. The three *_diagonal arms are the
# matched-severity readings of Section 6.5 and are not separate composites.
VOLUME_ARMS = ('arithmetic', 'geometric', 'rms', 'topsis',
               'arithmetic_weighted_SENSITIVITY')


def volume_arms():
    """Five composites x two models."""
    d = json.loads((EXP / 'pavia-composite-2026-07-25' / 'results-pavia-composite.json').read_text())
    rows = []
    for model, mv in d['models'].items():
        for arm in VOLUME_ARMS:
            dm = mv['arms'][arm]['delta_method']
            v = dm['cross_covariance_share_of_variance_pct']
            u = dm['understatement_of_width_pct']
            grad = dm['gradient_at_point']
            rows.append({
                'arm': f'volume/{model}/{arm}',
                'source': 'pavia-composite-2026-07-25/results-pavia-composite.json',
                'carries_gradient': True,
                'gradient_is_non_negative': all(g >= 0 for g in grad),
                'gradient_is_one_hot': sum(1 for g in grad if abs(g) > 1e-12) == 1,
                'cross_share_of_variance_pct': v,
                'implied_understatement_pct': u_from_v(v),
                'recorded_understatement_pct': u,
                'identity_residual_pp': u_from_v(v) - u,
                'cross_term_sum_is_zero': abs(v) < 1e-12,
            })
    return rows


def chain_arms():
    """Two German-credit chain arms and four real-system chain arms."""
    rows = []
    specs = [
        ('redraw-2026-07-24/results-redraw.json', 'german'),
        ('real-agentic-2026-07-25/results-real-agentic.json', 'real'),
    ]
    def emit(tag, label, node, rel):
        u = node.get('understatement_of_width_pct')
        if u is None:
            wm, wz = node.get('width_measured'), node.get('width_declared_zero')
            if wm is None or wz is None:
                return
            u = 100.0 * (1.0 - wz / wm)
        rows.append({
            'arm': f'chain/{tag}/{label}',
            'source': rel,
            'carries_gradient': True,
            'gradient_is_non_negative': True,
            'gradient_is_one_hot': False,
            'recorded_understatement_pct': u,
            'implied_variance_share_pct': 100.0 * (1.0 - (1.0 - u / 100.0) ** 2),
            'cross_term_sum_is_zero': abs(u) < 1e-12,
        })

    for rel, tag in specs:
        d = json.loads((EXP / rel).read_text())
        chain = d.get('chain', {})
        for k, v in chain.items():
            if not isinstance(v, dict):
                continue
            if 'width_measured' in v or 'understatement_of_width_pct' in v:
                emit(tag, k, v, rel)                      # german: fixed_draw / redraw
            else:
                for k2, v2 in v.items():                  # real: model -> scheme
                    if isinstance(v2, dict):
                        emit(tag, f'{k}/{k2}', v2, rel)
    return rows


def certificate_arms():
    """The two chain certificates. These carry a propagated interval and a
    zero-cross-term sensitivity, but no gradient of their own."""
    rows = []
    for name in ('german-chain.json', 'real-chain.json'):
        path = EXP / 'certificate-examples' / name
        if not path.exists():
            continue
        c = json.loads(path.read_text())['certificate']
        m = c['measurement']
        rows.append({
            'arm': f'certificate/{name[:-5]}',
            'source': f'certificate-examples/{name}',
            'carries_gradient': False,
            'gradient_is_one_hot': False,
            'recorded_understatement_pct': m['zero_cross_term_sensitivity']['width_change_pct'],
            'cross_term_sum_is_zero': False,
        })
    return rows


def main():
    rows = aggregation_arms() + volume_arms() + chain_arms() + certificate_arms()

    with_gradient = [r for r in rows if r['carries_gradient']]
    nonzero_cross = [r for r in with_gradient if not r['cross_term_sum_is_zero']]
    residuals = [abs(r['identity_residual_pp']) for r in rows if 'identity_residual_pp' in r]

    summary = {
        'study': 'Arm inventory behind the numerical-agreement remark',
        'total_arms': len(rows),
        'arms_carrying_a_gradient': len(with_gradient),
        'arms_with_a_non_zero_cross_term_sum': len(nonzero_cross),
        'arms_excluded_for_carrying_no_gradient':
            [r['arm'] for r in rows if not r['carries_gradient']],
        'arms_excluded_for_an_identically_zero_cross_term':
            [r['arm'] for r in with_gradient if r['cross_term_sum_is_zero']],
        'largest_identity_residual_pp': max(residuals) if residuals else None,
        'arms': rows,
    }
    OUT.write_text(json.dumps(summary, indent=1))

    print('total arms                        %d' % summary['total_arms'])
    print('carrying a gradient               %d' % summary['arms_carrying_a_gradient'])
    print('with a non-zero cross-term sum    %d' % summary['arms_with_a_non_zero_cross_term_sum'])
    print('excluded, no gradient             %s'
          % summary['arms_excluded_for_carrying_no_gradient'])
    print('excluded, cross-term identically zero:')
    for a in summary['arms_excluded_for_an_identically_zero_cross_term']:
        print('    %s' % a)
    if residuals:
        print('largest U-from-V identity residual  %.3g pp' % summary['largest_identity_residual_pp'])
    print('wrote %s' % OUT)


if __name__ == '__main__':
    main()
