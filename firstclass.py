class Test:
    def __init__(self, value):
        self.__val = value
    def __set_value(self, val2):
        self.__val = val2
    def __get_value(self):
        return self.__val
    val = property(__get_value, __set_value)
if __name__ == '__main__':
    obj = Test(60)
    print(obj.val)
    obj.val = 100
    print(obj.val)
    obj.val = 500
    print(obj.val)

