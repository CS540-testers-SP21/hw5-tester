'''
These tests are inspired by and use code from the tests made by cs540-testers
for the Fall 2020 semester

Their version can be found here: https://github.com/cs540-testers/hw8-tester/
'''

__maintainer__ = 'CS540-testers-SP21'
__author__ = ['Nicholas Beninato']
__credits__ = ['Harrison Clark', 'Stephen Jasina', 'Saurabh Kulkarni', 'Alex Moon']
__version__ = '1.0'

import unittest
import io
import sys
from time import time, sleep
from urllib.request import urlopen
import numpy as np
import numpy.testing
import regression

failures = []
errors = []
test_output = []

BODYFAT_FILE = 'bodyfat.csv'

def timeit(func):
    def timed_func(*args, **kwargs):
        global failures, errors
        t0 = time()
        try:
            out = func(*args, **kwargs)
            runtime = time() - t0
        except AssertionError as e:
            test_output.append(f'FAILED {func.__name__}')
            failures += [func.__name__]
            raise e
        except Exception as e:
            test_output.append(f'ERROR  {func.__name__}')
            errors += [func.__name__]
            raise e
        test_output.append(f'PASSED {func.__name__}{" "*(22-len(func.__name__))}in {(runtime)*1000:.2f}ms')
    return timed_func

class TestRegression(unittest.TestCase):
    @timeit
    def test1_get_dataset(self):
        dataset = regression.get_dataset(BODYFATE_FILE)
        self.assertEqual(dataset.shape, (252,16))

        self.assertEqual(dataset[0][0], 12.6)
        self.assertEqual(dataset[251][0], 30.7)
        self.assertEqual(dataset[75][7], 91.6)

    @timeit
    def test2_print_stats(self):
        dataset = regression.get_dataset(BODYFATE_FILE)
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        regression.print_stats(dataset, 1)
        output = capturedOutput.getvalue()
        sys.stdout = sys.__stdout__

        self.assertEqual(output, "252\n1.06\n0.02\n")

    @timeit
    def test3_regression(self):
        dataset = regression.get_dataset(BODYFATE_FILE)

        mse = regression.regression(dataset, cols=[2,3], betas=[0,0,0])
        numpy.testing.assert_almost_equal(mse, 418.50384920634923, 7)

        mse = regression.regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3])
        numpy.testing.assert_almost_equal(mse, 11859.17408611111, 7)

    @timeit
    def test4_gradient_descent(self):
        dataset = regression.get_dataset(BODYFATE_FILE)

        grad_desc = regression.gradient_descent(dataset, cols=[2,3], betas=[0,0,0])
        numpy.testing.assert_almost_equal(grad_desc, np.array([-37.87698413, -1756.37222222, -7055.35138889]))

    @timeit
    def test5_iterate_gradient(self):
        dataset = regression.get_dataset(BODYFATE_FILE)
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        regression.iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4)

        output = capturedOutput.getvalue()
        output_lines = output.split('\n')
        sys.stdout = sys.__stdout__

        expected_lines = ["1 423085332.40 394.45 -405.84 -220.18",
        "2 229744495.73 398.54 -401.54 163.14",
        "3 124756241.68 395.53 -404.71 -119.33",
        "4 67745350.04 397.75 -402.37 88.82",
        "5 36787203.39 396.11 -404.09 -64.57",
        "6 19976260.50 397.32 -402.82 48.47",
        "7 10847555.07 396.43 -403.76 -34.83",
        "8 5890470.68 397.09 -403.07 26.55",
        "9 3198666.69 396.60 -403.58 -18.68",
        "10 1736958.93 396.96 -403.20 14.65"]

        for out_line, exp_line in zip(output_lines, expected_lines):
            self.assertEqual(out_line.rstrip(), exp_line)

    @timeit
    def test6_compute_betas(self):
        dataset = regression.get_dataset(BODYFATE_FILE)
        betas = regression.compute_betas(dataset, [1,2])

        np.testing.assert_almost_equal(betas[0], 1.4029395600144443)
        np.testing.assert_almost_equal(betas[1], 441.3525943592249)
        np.testing.assert_almost_equal(betas[2], -400.5954953685588)
        np.testing.assert_almost_equal(betas[3], 0.009892204826346139)

    @timeit
    def test7_predict(self):
        dataset = regression.get_dataset(BODYFATE_FILE)
        prediction = regression.predict(dataset, cols=[1,2], features=[1.0708, 23])
        np.testing.assert_almost_equal(prediction, 12.62245862957813)

    @timeit
    def test9_synthetic_datasets(self):
        pass

    @timeit
    def test8_sgd():
        dataset = regression.get_dataset(BODYFATE_FILE)
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        regression.sgd(dataset, cols=[2,3], betas=[0,0,0], T=5, eta=1e-6)

        output = capturedOutput.getvalue()
        output_lines = output.split('\n')
        sys.stdout = sys.__stdout__

        expected_lines = ["1 387.33 0.00 0.00 0.00",
        "2 379.60 0.00 0.00 0.01",
        "3 335.99 0.00 0.00 0.01",
        "4 285.89 0.00 0.00 0.02",
        "5 245.75 0.00 0.01 0.03"]

        for out_line, exp_line in zip(output_lines, expected_lines):
            self.assertEqual(out_line.rstrip(), exp_line)

def get_versions():
    current = __version__
    to_tuple = lambda x: tuple(map(int, x.split('.')))
    try:
        with urlopen('https://raw.githubusercontent.com/CS540-testers-SP21/hw5-tester/master/.version') as f:
            if f.status != 200:
                raise Exception
            latest = f.read().decode('utf-8')
    except Exception as e:
        print('Erorr checking for latest version') # very descriptive error messages
        return to_tuple(current), to_tuple(current) # ignoring errors probably isn't the best idea tbh
    return to_tuple(current), to_tuple(latest)

if __name__ == '__main__':
    print(f'Running CS540 SP21 HW5 tester v{__version__}\n')

    current, latest = get_versions()
    to_v_str = lambda x : '.'.join(map(str, x))
    if current < latest:
        print(f'A newer version of this tester (v{to_v_str(latest)}) is available. You are current running v{to_v_str(current)}\n')
        print('You can download the latest version at https://github.com/CS540-testers-SP21/hw5-tester\n')
    
    unittest.main(argv=sys.argv, exit=False)
    sleep(.01)
    for message in test_output:
        print(message)
    print()
    if not failures and not errors:
        print('\nPassed all tests successfully\n')
    if failures:
        print('The following tests failed:\n' + '\n'.join(failures) + '\n')
    if errors:
        print('The following tests had exceptions when running:\n' + '\n'.join(errors) + '\n')
    if failures or errors:
        print('Please see the Traceback above for where there were issues')
