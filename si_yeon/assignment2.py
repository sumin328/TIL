## 0830
# 2. class_scores 총합, 평균 (수업 Python #03)
# 총합
class_scores = [
    [30, 90, 80, 40, 50],
    [100, 100, 100, 100, 100],
    [90, 90, 30, 30, 20]
]

'''
class_scores[0]
class_scores[1]
class_scores[2]
'''
total_score = 0
'''
for i in class_scores[0]:  #class_scores[0] = [30, 90, 80, 40, 50]
    total_score += i
print(total_score)
'''
for j in range(0, len(class_scores)): #range(0, len(class_scores)) = range(0, 3) = [0, 1, 2]
    for i in class_scores[j]:  #class_scores[0] = [30, 90, 80, 40, 50]
        total_score += i

print(total_score)

# 평균

subjects_count = 0
'''
len(class_scores) #3

len(class_scores[0])
len(class_scores[1])
len(class_scores[2])
'''
for i in range(len(class_scores)):  # range(0,3) : 0,1,2
    subjects_count += len(class_scores[i])

print(subjects_count)

average_score = total_score/subjects_count
print(average_score)


