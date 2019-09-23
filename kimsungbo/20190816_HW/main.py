from grades import grade
import pandas as pd

class_scores = [
    {
        '국어': 80,
        '영어': 100,
        '수학': 50
    },
    {
        ''
        ''
        '국어': 90,
        '영어': 70,
        '수학': 40
    }
]

total = {}

for subject in ['국어', '영어', '수학']:
    total[subject] = grade(class_scores, subject)
    print(total[subject])

print(total)

table = pd.DataFrame(total, columns=['korean', 'english', 'math'])
print(table)
