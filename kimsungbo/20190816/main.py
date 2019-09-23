import pandas as pd

from scores import class_total


def main():
    class_scores = [
        {'Name': 'Jane', 'Korean': 100, 'English': 100, 'Math': 100},
        {'Name': 'Brown', 'Korean': 10, 'English': 20, 'Math': 30}
    ]

    total = {'Name': '<Total>'}
    for subject in ['국어', '영어', '수학']:
        total[subject] = class_total(class_scores, subject)

    table = pd.DataFrame(class_scores + [total],
                         columns=['Name', 'Korean', 'English', 'Math'])
    print(table)
