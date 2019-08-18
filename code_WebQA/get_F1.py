import sys
sys.path.append("../")
sys.path.append("../evaluation")
from evaluation.fuzzy_matching import FuzzyMatcher

# fuzzy_macher = FuzzyMatcher()
# print(fuzzy_macher.is_synonym("黑色", "黑色的猩猩"))

class F1Counter:
    def __init__(self, is_fuzzy=False):
        self.total = 0
        self.correct = 0
        self.fuzzy_macher = FuzzyMatcher() if is_fuzzy else None
        self.result_dict = dict()  # 保存序号即是否正确，格式： {1:True, 2:False, ...}

    def update(self, i, glod_real, pre_real):
        self.total += 1
        if self.fuzzy_macher:
            is_correct = self.fuzzy_macher.is_synonym(glod_real, pre_real)
        else:
            is_correct = glod_real == pre_real
        self.result_dict[i] = is_correct
        if is_correct:
            self.correct += 1
        return is_correct

    def get_f1(self):
        f1 = self.correct/self.total if self.total != 0 else 0
        return self.total, self.correct, f1


