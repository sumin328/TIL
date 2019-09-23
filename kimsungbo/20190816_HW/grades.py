def grade(scores, subject):
    grades = []
    for score in scores:
        score = score.get(subject)
        if score > 80:
            grades.append('A')
        elif score > 60:
            grades.append('B')
        elif score > 40:
            grades.append('C')
        elif score > 20:
            grades.append('D')
        else:
            grades.append('F')
    return grades
