"""
Make this repository's `utils` directory a proper Python package.

Some test suites import modules via `from utils.xxx import ...`. Without an
`__init__.py`, Python may resolve `utils` to an unrelated third-party or
sibling repository's package found earlier on `sys.path`.
"""

__all__ = []

