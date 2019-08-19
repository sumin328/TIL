'''
file

import : import library (eg. import pandas as pd)

from - import : import function/ data thing from somewhere (eg. import scores from 20190819.py)

library: pre-made codes

package: pre-made library is shared as a package (eg. pandas, numpy)

pip install: install packages


'''


def less_number(x, y):
    if x < y:
        return x
    return y


def select_high_scores(scores, pivot):
    # Initial accumulator
    high_scores = []

    # Accumulation
    for score in scores:
        # Conditional
        if score >= pivot:
            high_scores.append(score)

    # Return
    return high_scores

