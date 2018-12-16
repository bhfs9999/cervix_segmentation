import numpy as np

list_num = 6
true_num = 5

class Comparator(object):
    def __init__(self, lr, auto_switch, op, decay_list, decay_ratio=0.1, max_decay=3):
        self.lr = lr
        self.decay_list = decay_list
        self.compare_op = op
        self.decay_ratio = decay_ratio
        self.auto_switch = auto_switch
        if not self.auto_switch:
            assert isinstance(self.decay_list, list)
        else:
            print('Now, using auto learning decay strategy!')
            self.result_que = [False] * list_num
            self.max_decay = max_decay
            self.decay_num = 0
            if op == 'larger_than':
                self.best_value, self.last_value = 0, 0
            elif op == 'less_than':
                self.best_value, self.last_value = 1000, 1000
            else:
                print('only "larger_than" or "less_than" supported!')
                raise ValueError

    def equal(self, new_value):
        return (np.array(self.decay_list) == new_value).sum() == 1

    def larger_than(self, new_value, metric):
        return new_value > metric

    def less_than(self, new_value, metric):
        return new_value < metric

    def re_init(self):
        if self.compare_op=='larger_than':
            self.last_value = 0
        elif self.compare_op == 'less_than':
            self.last_value = 1000
        else:
            print('only "larger_than" or "less_than" supported!')
            raise ValueError
        self.result_que = [False] * list_num

    def __call__(self, new_value):
        if not self.auto_switch:
            if self.equal(new_value):
                self.lr = self.lr * self.decay_ratio
                return 'decay'
            else:
                return 'invariable'
        else:
            best_tag = getattr(self, self.compare_op)(new_value, self.best_value)
            if best_tag:
                self.best_value = new_value
                self.last_value = new_value
                self.result_que = [False] * list_num
                return 'invariable'
            last_tag = getattr(self, self.compare_op)(new_value, self.last_value)
            if last_tag:
                pass
            else:
                self.result_que.pop(0)
                self.result_que.append(True)
            self.last_value = new_value
            if self.result_que.count(True) >= true_num:
                self.re_init()
                self.decay_num += 1
                self.lr = self.lr * self.decay_ratio
                if self.decay_num <= self.max_decay:
                    return 'decay'
                else:
                    # return 'early_stop'
                    return 'invariable'
            return 'invariable'

