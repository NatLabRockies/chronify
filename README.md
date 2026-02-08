# chronify

[![Documentation](https://img.shields.io/badge/docs-ready-blue.svg)](https://nrel.github.io/chronify)
[![codecov](https://codecov.io/gh/nrel/chronify/graph/badge.svg?token=WIY2KAOX63)](https://codecov.io/gh/nrel/chronify)


This package implements a store for time series data in support of Python-based
modeling packages. It supports validation and mapping across different time configurations.

## Package Developer Guide
🚧

## Installation
To use DuckDB or SQLite as the backend:
```
$ pip install chronify
```

To use Apache Spark via Apache Thrift Server as the backend:
```
$ pip install "chronify[spark]"
```

## Developer installation
```
$ pip install -e ".[dev,spark]"
```

Please install `pre-commit` so that your code is checked before making commits.
```
$ pre-commit install
```

## License
chronify is developed under NREL Software Record SWR-21-52, "demand-side grid model".
[License](https://github.com/NREL/chronify/blob/main/LICENSE).
