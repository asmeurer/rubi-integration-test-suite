# rubi-integration-test-suite

The Rubi *MathematicaSyntaxTestSuite* translated to SymPy: 66k+ integral test
cases as `RubiTestSuiteCase` objects (integrand, variable, expected
antiderivative, Rubi step count), plus the generator that produced them and a
runner that drives the `rubi-integrate` package over the corpus.

> **⚠️ Experimental** — this package is under active development; layout and
> content may change without notice. Version 0.0.1 is a pre-alpha snapshot and
> has not been published to PyPI yet.

- `rubi_integration_test_suite/` — the generated corpus modules
- `generator_for_rubi_test_suite.py` — regenerates them from the Mathematica
  test-suite sources (needs the `generator` extra: `sympy-wolfram`)
- `rubi_test_suite_runner.py` — runs `rubi-integrate` over the corpus (needs
  the `runner` extra: `rubi-integrate`)

## License

MIT (Copyright (c) 2026 Francesco Bonazzi) for the Python port. The Rubi-derived
content (rules / test corpus) originates from [Rubi](https://rulebasedintegration.org)
by Albert Rich, whose MIT license is reproduced in full in `LICENSE`.
