# 190919 Python Study 

Created By: 보경 정
Last Edited: Sep 22, 2019 10:46 PM

# Even the last

- 파이썬에서는 마이너스 인덱싱이 가능
`last = array[len(array)-1]` == `last = array[-1]`

# Secret Message

- 텍스트에 대한 for문 돌릴 때 i 말고 char 등의 조금 더 식별이 쉬운 이름으로 작성하기
- isupper()는 true, false 값을 반환하기 때문에 ==True를 붙일 필요가 없음 → 그 자체를 조건으로 두어도 True면 다음 연산으로 넘어감

# Three words

- 조건문은 true, false로 값이 나오기 때문에 'return 조건문'으로 하는 것도 가능

# Index Power

- len(array) 범위 다시 생각해보기

# 체육복

- pass, continue의 차이
    - pass : 실행한 연산은 없으나 한 줄을 채워야 할 때 사용. 다음 줄에 같은 인덴트의 코드가 있다면 그 코드를 실행함.
    - continue : 같은 블럭 내의 이후 코드를 모두 건너뛴다. 바로 반복문의 처음으로 돌아감.
- dir(list) : list 내의 method 확인 가능
- 사용법을 잘 모르겠다면 help로 사용법 알아보기