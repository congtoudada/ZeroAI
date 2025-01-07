from typing import Dict


class ObjectPool:
    def __init__(self, capacity: int, class_type):
        if capacity <= 0:
            raise ValueError('capacity must be greater than 0')
        self.class_type = class_type
        self.origin_capacity = capacity
        self.pool = []
        self._create(self.origin_capacity)

    def _create(self, size):
        for i in range(size):
            if self.class_type == dict:
                self.pool.append({})
            elif self.class_type == list:
                self.pool.append([])
            else:
                self.pool.append(self.class_type())

    def pop(self):
        if self.pool.__len__() <= 0:
            create_num = int(self.origin_capacity * 0.5)
            print(f"Object Pool is shortage, will create {create_num}")
            self._create(create_num)

        return self.pool.pop()

    def push(self, obj):
        # 最大不超过origin_capacity的2倍
        if self.get_size() > self.origin_capacity * 2:
            return
        if isinstance(obj, self.class_type):
            self.pool.append(obj)

    def clear(self):
        self.pool.clear()

    def set_origin_capacity(self, capacity: int):
        self.origin_capacity = capacity

    def get_origin_capacity(self):
        return self.origin_capacity

    def get_size(self):
        return self.pool.__len__()


class ObjectPoolItem:
    def __init__(self):
        self.val_int = 100
        self.val_str = "cong"


if __name__ == '__main__':
    # 自定义对象测试
    pool = ObjectPool(3, ObjectPoolItem)
    print(pool.get_size())
    item = pool.pop()

    print(pool.get_size())
    item.val_int = 200
    pool.push(item)
    print(pool.get_size())
    item = pool.pop()  # 取的是同一个对象
    print(item.val_int)

    for i in range(4):
        ret = pool.pop()
    print(pool.get_size())

    # 字典测试
    pool = ObjectPool(3, dict)
    print(pool.get_size())
    item = pool.pop()
    item["val_int"] = 100
    print(pool.get_size())
    pool.push(item)
    print(pool.get_size())
    item = pool.pop()
    print(item["val_int"])
    print(pool.get_size())


