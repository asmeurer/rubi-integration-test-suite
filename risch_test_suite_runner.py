# -*- coding: utf-8 -*-
"""Run sympy's risch_integrate over the Rubi test-suite corpus.

If the sympy under test has the experimental algebraic=True
(exp-log-tower) support, radicals are attempted through it; otherwise
this measures stock risch_integrate.

Usage:
    python risch_test_suite_runner.py <sympy-repo-path> <subpackage> [limit]

e.g.
    python risch_test_suite_runner.py ../sympy \
        t_1_algebraic_functions.t_1_1_binomial_products.t_1_1_1_linear

Cases are filtered to those risch_integrate can even attempt: every
exponent appearing in the integrand must be a concrete Rational (Rubi's
corpus is heavily parameterized by symbolic exponents m, n, p, which no
Risch-style algorithm handles).  Cases with symbolic *constants* are
kept but reported separately from fully concrete ones.

Classification per case:
  SOLVED    -- no unevaluated Integral in the result (for algebraic
               towers the result was certified internally by the
               tower-level acceptance filter)
  partial   -- result contains an unevaluated plain Integral
  CLAIMS-NE -- result contains a NonElementaryIntegral (should never
               happen for algebraic towers: a bug if it does)
  NIE       -- NotImplementedError
  timeout   -- exceeded the per-case time limit
  error:*   -- other exception
"""
import importlib
import inspect
import json
import os
import pkgutil
import signal
import sys
import time
from collections import Counter

sys.path.insert(0, sys.argv[1])          # the sympy checkout to test
sys.path.insert(0, '.')                   # this repo (corpus package)

from sympy import (Integral, Rational, Pow, nan, oo, zoo, integrate,  # noqa: E402
    latex)
from sympy.integrals.risch import (risch_integrate,      # noqa: E402
    NonElementaryIntegral)

import rubi_integration_test_suite as corpus_root  # noqa: E402

TIMEOUT = 5

# The experimental exp-log-tower support for radicals (sympy branch
# risch-algebraic); without it this measures stock risch_integrate.
HAS_ALGEBRAIC = 'algebraic' in inspect.signature(risch_integrate).parameters
RISCH_KWARGS = {'algebraic': True} if HAS_ALGEBRAIC else {}

# Environment knobs (so the CLI stays simple):
#   RISCH_RESULTS       append one JSON line per case to this path
#   RISCH_MODE          'algebraic' (default: radical cases) or
#                       'transcendental' (radical-free exp-log cases,
#                       run without the algebraic towers)
#   RISCH_HANDLE_FIRST  'log' (default) or 'exp' (tower order)
RESULTS_PATH = os.environ.get('RISCH_RESULTS')
MODE = os.environ.get('RISCH_MODE', 'algebraic')
HANDLE_FIRST = os.environ.get('RISCH_HANDLE_FIRST', 'log')
if MODE == 'transcendental':
    RISCH_KWARGS = {}
RISCH_KWARGS['handle_first'] = HANDLE_FIRST


def iter_cases(subpackage):
    pkg = importlib.import_module('rubi_integration_test_suite.' + subpackage)
    if hasattr(pkg, 'TEST_CASES'):
        # a single leaf module was named directly
        for case in pkg.TEST_CASES:
            yield pkg.__name__, case
        return
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        if modinfo.ispkg:
            continue
        mod = importlib.import_module(modinfo.name)
        for case in getattr(mod, 'TEST_CASES', []):
            yield modinfo.name, case


def attemptable(f, x):
    # every exponent must be a concrete Rational (integer exponents are
    # fine too); reject symbolic exponents outright
    for p in f.atoms(Pow):
        if not (p.exp.is_Rational or not p.exp.free_symbols):
            return False
        if not p.exp.is_Rational:
            return False
    if f.has(nan, oo, zoo):
        return False
    # a "true" radical for the algebraic towers: fractional power of an
    # x-dependent base that is not itself an exponential (sqrt(exp(u))
    # rewrites to exp(u/2), and constant radicals just extend the
    # constant field -- both leave the tower transcendental)
    has_radical = any(p.exp.is_Rational and not p.exp.is_Integer
                      and p.base.has(x) and not p.base.func.__name__ == 'exp'
                      for p in f.atoms(Pow))
    if MODE == 'transcendental':
        return not has_radical
    return has_radical


def main():
    subpackage = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    signal.signal(signal.SIGALRM,
                  lambda s, fr: (_ for _ in ()).throw(TimeoutError()))

    stats = {'concrete': Counter(), 'parametric': Counter()}
    interesting = []            # (kind, module, integrand, classification)
    n_seen = n_tried = 0
    t0 = time.time()
    for modname, case in iter_cases(subpackage):
        n_seen += 1
        f, x = case.integrand, case.variable
        if not attemptable(f, x):
            continue
        kind = 'concrete' if f.free_symbols <= {x} else 'parametric'
        reason = ''
        n_tried += 1
        if limit and n_tried > limit:
            n_tried -= 1
            break
        t_case = time.time()
        signal.alarm(TIMEOUT)
        try:
            r = risch_integrate(f, x, **RISCH_KWARGS)
            if isinstance(r, NonElementaryIntegral) or \
                    r.has(NonElementaryIntegral):
                cls = 'CLAIMS-NE'
            elif r.has(Integral):
                cls = 'partial'
            else:
                cls = 'SOLVED'
        except TimeoutError:
            cls = 'timeout'
            print('  TIMEOUT-CASE %s | %s' % (modname.rsplit('.', 1)[-1], f),
                  flush=True)
        except NotImplementedError as e:
            cls = 'NIE'
            reason = str(e)[:160].replace('\n', ' ')
        except Exception as e:
            cls = 'error:' + type(e).__name__
        finally:
            signal.alarm(0)
        if cls == 'SOLVED':
            # Is this a NEW capability, or could plain integrate()
            # (which does not use the algebraic towers) already do it?
            signal.alarm(2*TIMEOUT)
            try:
                ri = integrate(f, x, risch=False)
                old_ok = not ri.has(Integral)
            except Exception:
                old_ok = False
            finally:
                signal.alarm(0)
            cls = 'SOLVED-both' if old_ok else 'SOLVED-NEW'
        stats[kind][cls] += 1
        if RESULTS_PATH:
            with open(RESULTS_PATH, 'a') as fh:
                fh.write(json.dumps({'mod': modname.rsplit('.', 1)[-1],
                    'expr': str(f), 'latex': latex(f), 'kind': kind,
                    'cls': cls, 'reason': reason,
                    'secs': round(time.time() - t_case, 3)}) + '\n')
        if n_tried % 50 == 0:
            print('  ...%d tried, %.0f s' % (n_tried, time.time() - t0),
                  flush=True)
        if cls in ('SOLVED-NEW', 'CLAIMS-NE') or cls.startswith('error'):
            interesting.append((kind, modname.rsplit('.', 1)[-1], f, cls))

    dt = time.time() - t0
    print('subpackage: %s' % subpackage)
    print('cases seen: %d, attemptable radicals tried: %d, %.0f s'
          % (n_seen, n_tried, dt))
    for kind in ('concrete', 'parametric'):
        tot = sum(stats[kind].values())
        if tot:
            print('%-11s (%4d): %s' % (kind, tot,
                  ', '.join('%s %d' % kv
                            for kv in stats[kind].most_common())))
    print()
    for kind, mod, f, cls in interesting[:60]:
        print('  %-10s %-11s %-12s %s' % (cls, kind, mod, f))


if __name__ == '__main__':
    main()
