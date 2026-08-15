# -*- coding: utf-8 -*-
"""Numerical oracle comparing risch_integrate results against the
corpus' expected antiderivatives.

The primary verdict is numerical and never trusts symbolic
zero-testing of radical identities: it differentiates our answer and
compares against the integrand pointwise, at points chosen to cover
both signs of every factor appearing under a radical (a branch error
is invisible if you only sample where the radicands are positive).
The expected antiderivative is used only for secondary taxonomy, so a
mistranslated or unevaluatable corpus answer cannot corrupt the
verdict on our answer.

Per-instantiation verdicts:

  DERIV-OK         D(ours) == f at every usable sample point, and
                   ours - expected is one constant across all regions
  DERIV-OK-SPLIT   D(ours) == f everywhere, but ours - expected is a
                   different constant on different connected regions
                   (form or branch-convention difference; both
                   antiderivatives correct where checked)
  DERIV-OK-EXP-BAD D(ours) == f everywhere, and D(expected) != f at
                   some usable point (the corpus answer itself looks
                   wrong there)
  DERIV-OK-EXP-NC  D(ours) == f everywhere; expected could not be
                   compared (failed to evaluate, or not closed form)
  WRONG            D(ours) != f at a sample point, confirmed at double
                   precision -- our answer is wrong there
  UNDECIDED-*      not enough usable sample points, or numerics were
                   inconclusive

For parametric cases the check runs once per instantiation round
(positive / mixed-sign / negative / irrational / complex constants)
and the aggregate verdict is the worst per-round outcome, where
WRONG > UNDECIDED-* > DERIV-OK-EXP-BAD > DERIV-OK-SPLIT > ... .
"""
import signal
from bisect import bisect

from sympy import (I, Integral, Mul, Pow, Rational, S, im, nan, oo, re,
                   sqrt, zoo)
from sympy.core.symbol import Symbol

__all__ = ['check_case']

#: relative tolerance at working precision for "equal"
EQ_TOL = Rational(1, 10)**25
#: relative tolerance at double precision below which a first-pass
#: mismatch is written off as a precision artifact
EQ_TOL2 = Rational(1, 10)**40
#: relative tolerance above which a double-precision recheck confirms
#: a genuine mismatch
NEQ_TOL = Rational(1, 10)**12
PREC = 60

# Verdict severity, worst first, for aggregating instantiation rounds.
# TIMEOUT outranks the OK verdicts: an OK aggregate must mean every
# applicable round was actually checked.  DEGENERATE stays at the
# bottom (a round skipped as inapplicable does not taint an OK case).
SEVERITY = ['WRONG', 'UNDECIDED-NUMERICS', 'UNDECIDED-COVERAGE',
            'TIMEOUT', 'DERIV-OK-EXP-BAD', 'DERIV-OK-SPLIT',
            'DERIV-OK-EXP-NC', 'DERIV-OK', 'DEGENERATE']


def _radicand_bases(f, x):
    """x-dependent bases raised to non-integer rational powers in f."""
    bases = []
    for p in f.atoms(Pow):
        if (p.exp.is_Rational and not p.exp.is_Integer and p.base.has(x)
                and not p.base.is_Symbol):
            bases.append(p.base)
        elif p.exp.is_Rational and not p.exp.is_Integer and p.base == x:
            bases.append(p.base)
    return bases


def _dyadic(v, bits=24):
    """A nearby dyadic rational (never exactly a nondyadic root)."""
    return Rational(int(round(v * 2**bits)), 2**bits)


def _real_breakpoints(bases, x):
    """Sorted real roots/poles of the radicand bases, as floats."""
    from sympy import Poly, together
    pts = set()
    for b in bases:
        b = together(b)
        n, d = b.as_numer_denom()
        for part in (n, d):
            if not part.has(x):
                continue
            try:
                p = Poly(part, x)
            except Exception:
                continue
            if p.degree() < 1 or not p.domain.is_Exact or \
                    not (p.domain.is_QQ or p.domain.is_ZZ):
                try:
                    rts = [r for r in p.nroots() if abs(im(r)) < 1e-12]
                    pts.update(float(re(r)) for r in rts)
                except Exception:
                    pass
                continue
            try:
                for r in p.real_roots():
                    pts.add(float(r.evalf(30)))
            except Exception:
                continue
    return sorted(pts)


def _sample_points(breaks):
    """Real dyadic sample points around the breakpoints, plus a default
    grid and a few complex points."""
    real = set()
    default = [Rational(p, 64) for p in
               (-333, -173, -87, -29, 23, 91, 169, 351)]
    if breaks:
        ext = [breaks[0] - 2.0] + breaks + [breaks[-1] + 2.0]
        for a, b in zip(ext, ext[1:]):
            if b - a > 1e-9:
                real.add(_dyadic((a + b) / 2))
        for r in breaks:
            gap = min(x for x in
                      [1.0] + [abs(r - s) for s in breaks if s != r])
            d = gap / 4
            real.add(_dyadic(r - d))
            real.add(_dyadic(r + d))
    real.update(default)
    cx = [Rational(1, 2) + 3*I/4, Rational(-5, 4) + I/3,
          2 - I/2, Rational(-3, 4) - 5*I/4]
    return sorted(real), cx


def _eval_at(expr, x, pt, prec=PREC):
    """expr at x=pt as an (re, im) pair of Floats, or None."""
    try:
        v = expr.subs(x, pt).evalf(prec)
    except Exception:
        return None
    if v.has(nan, oo, zoo, Integral):
        return None
    if v.free_symbols:
        return None
    try:
        vr, vi = v.as_real_imag()
        vr, vi = vr.evalf(prec), vi.evalf(prec)
        if not (vr.is_Number and vi.is_Number):
            return None
        float(vr), float(vi)  # reject non-finite
    except (TypeError, ValueError, OverflowError, Exception):
        return None
    return (vr, vi)


def _absval(v):
    return (v[0]**2 + v[1]**2)


def _reldiff2(a, b):
    """Squared relative difference of two (re, im) pairs."""
    d2 = (a[0] - b[0])**2 + (a[1] - b[1])**2
    scale = max(S.One, _absval(a), _absval(b))
    return d2 / scale


def _match_at(e1, e2, x, pt, prec=PREC):
    """'eq' / 'neq' / 'skip' / 'undecided' for e1 == e2 at x=pt."""
    a, b = _eval_at(e1, x, pt, prec), _eval_at(e2, x, pt, prec)
    if a is None or b is None:
        return 'skip'
    if _reldiff2(a, b) < EQ_TOL**2:
        return 'eq'
    a2, b2 = _eval_at(e1, x, pt, 2*prec), _eval_at(e2, x, pt, 2*prec)
    if a2 is None or b2 is None:
        return 'undecided'
    r2 = _reldiff2(a2, b2)
    if r2 < EQ_TOL2**2:
        return 'eq'
    if r2 > NEQ_TOL**2:
        return 'neq'
    return 'undecided'


def check_answer(f, x, ours, expected):
    """Run the numerical ladder for one (already concrete) case.

    Returns a dict with 'verdict' and supporting detail.
    """
    from sympy import diff
    dours = diff(ours, x)
    bases = _radicand_bases(f, x)
    breaks = _real_breakpoints(bases, x)
    real_pts, cx_pts = _sample_points(breaks)

    used, undecided = [], []
    for pt in real_pts + cx_pts:
        m = _match_at(dours, f, x, pt)
        if m == 'skip':
            continue
        if m == 'neq':
            return {'verdict': 'WRONG', 'point': str(pt),
                    'breaks': [round(b, 6) for b in breaks]}
        if m == 'undecided':
            undecided.append(pt)
            continue
        used.append(pt)

    # Coverage: every radicand with real roots/poles must have been
    # checked on both sides of at least one of them (the branch-bug
    # class flips sign at those points and is untested otherwise).
    cover_ok = True
    real_used = [p for p in used if p.is_real]
    for b in bases:
        bks = _real_breakpoints([b], x)
        if not bks:
            continue
        if not any(any(float(p) < r for p in real_used) and
                   any(float(p) > r for p in real_used) for r in bks):
            cover_ok = False

    if len(used) < 4 or not cover_ok:
        return {'verdict': 'UNDECIDED-COVERAGE', 'used': len(used),
                'undecided': len(undecided)}
    if undecided:
        return {'verdict': 'UNDECIDED-NUMERICS', 'used': len(used),
                'undecided': len(undecided)}

    # ours is right; classify against expected (real points only --
    # region structure in C is not an interval partition).
    if expected is None or expected.has(Integral):
        return {'verdict': 'DERIV-OK-EXP-NC', 'used': len(used)}
    consts = {}
    for pt in real_used:
        a = _eval_at(ours, x, pt)
        b = _eval_at(expected, x, pt)
        if a is None or b is None:
            continue
        region = bisect(breaks, float(pt))
        consts.setdefault(region, []).append(
            (a[0] - b[0], a[1] - b[1]))
    if not consts:
        return {'verdict': 'DERIV-OK-EXP-NC', 'used': len(used)}

    def _cdiff(u, v):
        d2 = (u[0] - v[0])**2 + (u[1] - v[1])**2
        scale = max(S.One, u[0]**2 + u[1]**2, v[0]**2 + v[1]**2)
        return d2 / scale

    local_const = True
    reps = []
    for region, vals in sorted(consts.items()):
        for v in vals[1:]:
            if _cdiff(vals[0], v) > EQ_TOL2:
                local_const = False
        reps.append(vals[0])
    if not local_const:
        # ours - expected is not constant within a region.  Either the
        # corpus answer is wrong there, or the region structure above
        # (radicand breakpoints only) is too coarse for expected's own
        # jumps (atan/log arguments).  Discriminate rigorously: check
        # D(expected) against f directly.
        dexp = diff(expected, x)
        for pt in real_used:
            m = _match_at(dexp, f, x, pt)
            if m == 'neq':
                return {'verdict': 'DERIV-OK-EXP-BAD', 'point': str(pt),
                        'used': len(used)}
        return {'verdict': 'DERIV-OK-SPLIT', 'used': len(used),
                'coarse': True}
    for v in reps[1:]:
        if _cdiff(reps[0], v) > EQ_TOL2:
            return {'verdict': 'DERIV-OK-SPLIT', 'used': len(used),
                    'nregions': len(reps)}
    return {'verdict': 'DERIV-OK', 'used': len(used)}


#: instantiation pools, cycled over the sorted constant symbols
ROUNDS = [
    ('pos', [2, 3, 5, 7, 11]),
    ('mixed', [-2, 3, Rational(-1, 2), 5, Rational(-7, 3)]),
    ('neg', [-3, Rational(-5, 2), -7, -2, Rational(-1, 3)]),
    ('irr', [sqrt(2), -sqrt(3), 1 + sqrt(2), 2, Rational(1, 2)]),
    ('cx', [1 + I/2, 2, -1 + I, 3, I/3]),
]


def _degenerate(f_i):
    from sympy import together
    if f_i.has(nan, oo, zoo):
        return True
    try:
        return together(f_i).is_zero is True
    except Exception:
        return True


def check_case(f, x, ours, expected, timeout=60):
    """Check one corpus case; instantiate symbolic constants if present.

    Returns {'verdict': aggregate, 'rounds': {name: perround}} where
    perround are check_answer() dicts.  A per-round signal.alarm(...)
    guards each instantiation; TimeoutError is recorded, not raised
    (a SIGALRM handler raising TimeoutError must already be installed
    when timeout is nonzero).
    """
    consts = sorted(f.free_symbols - {x}, key=lambda s: s.name)
    if not consts:
        plan = [('concrete', {})]
    else:
        plan = []
        for name, pool in ROUNDS:
            sub = {s: pool[i % len(pool)] for i, s in enumerate(consts)}
            plan.append((name, sub))
    rounds = {}
    for name, sub in plan:
        if timeout:
            signal.alarm(timeout)
        try:
            f_i = f.subs(sub) if sub else f
            if sub and _degenerate(f_i):
                rounds[name] = {'verdict': 'DEGENERATE'}
                continue
            ours_i = ours.subs(sub) if sub else ours
            exp_i = expected.subs(sub) if (sub and expected is not None) \
                else expected
            if sub and (ours_i.has(nan, oo, zoo) or
                        exp_i is not None and exp_i.has(nan, oo, zoo)):
                rounds[name] = {'verdict': 'DEGENERATE'}
                continue
            rounds[name] = check_answer(f_i, x, ours_i, exp_i)
        except TimeoutError:
            rounds[name] = {'verdict': 'TIMEOUT'}
        except Exception as e:
            rounds[name] = {'verdict': 'UNDECIDED-NUMERICS',
                            'error': type(e).__name__}
        finally:
            if timeout:
                signal.alarm(0)
    agg = min((r['verdict'] for r in rounds.values()),
              key=lambda v: SEVERITY.index(v) if v in SEVERITY else 99)
    return {'verdict': agg, 'rounds': rounds}
