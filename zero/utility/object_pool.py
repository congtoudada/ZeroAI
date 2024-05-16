class ObjectPool:
    def __init__(self, capacity: int, class_type):
        self.class_type = class_type
        self.origin_capacity = capacity
        self.pool = []
        self._init(self.origin_capacity)

    def _init(self, capacity):
        for i in range(capacity):
            self.pool.append(self.class_type())

    def pop(self):
        if self.pool.__len__() <= 0:
            self._init(int(self.origin_capacity * 0.5))

        return self.pool.pop()

    def push(self, obj):
        if isinstance(obj, type(self.class_type)):
            self.pool.append(obj)

    def clear(self):
        self.pool.clear()

    def set_origin_capacity(self, capacity: int):
        self.origin_capacity = capacity

    def get_origin_capacity(self):
        return self.origin_capacity