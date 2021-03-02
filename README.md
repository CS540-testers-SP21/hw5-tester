# hw5-tester

Tests for CS540 Spring 2021 HW5: Linear Regression 

# The tester is still under development. There will be an announcement on Discord once a working version is released

## Usage

Download [test.py](test.py) move it into the directory that contains `regression.py` and `bodyfat.csv`

The contents of your directory should look like this:

```shell
$ tree
.
├── regression.py
├── bodyfat.csv
└── test.py
```

To run the tests, do

```python
$ python3 test.py
```

Ideally, you should be running `test.py` using your terminal as this README describes. If you have an issue, first try running it that way. However, provided that `test.py`, `regression.py`, and `bodyfat.csv` are all in the same directory, it should work if you do `%run test.py` in Jupyter, or run it the same way you would run `regression.py` in your editor (VS Code, Pycharm, Sublime, etc).

### These tests _do not_ check for `plot_mse`

## Disclaimer

These tests are not endorsed or created by anyone working in an official capacity with UW Madison or any staff for CS540. The tests are make by students, for students.

By running `test.py`, you are executing code you downloaded from the internet. Back up your files and take a look at what you are running first.

If you have comments or questions, create an issue at [https://github.com/CS540-testers-SP21/hw3-tester/issues](https://github.com/CS540-testers-SP21/hw4-tester/issues) or ask in our discord at [https://discord.gg/RDFNsAxgCQ](https://discord.gg/RDFNsAxgCQ).
