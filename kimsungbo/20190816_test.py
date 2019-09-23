from py20190816 import less_number
from py20190816 import select_high_scores


def test_less_number():
    assert less_number(1, 2) == 1
    assert less_number(2, 1) == 1


def test_select_high_scores():
    assert select_high_scores([60], 50) == [60]