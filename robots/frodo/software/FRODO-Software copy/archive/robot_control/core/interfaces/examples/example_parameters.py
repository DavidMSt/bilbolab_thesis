from core.device.parameters import DataLink

a = 3


class Testclass:
    x: float
    y: int
    name: str

    def __init__(self, x, y, name):
        self.x = x
        self.name = name
        self.y = y

    def x_setter(self, value):
        self.x = value * 2
        return True


def main():
    global a
    testobject = Testclass(3, 2, 'Testobject')

    values = {
        'a': 3
    }

    _params1 = {
        'param1': DataLink(identifier='param1', description='This is a test parameter', datatype=float, limits=[0, 20],
                           obj=testobject, name='x', write_function=testobject.x_setter),
        'param2': DataLink(identifier='param2', description='Second Parameter', datatype=int, obj=testobject, name='y'),
        'param3': DataLink(identifier='param3', description='Third Parameter', datatype=str, obj=testobject, name='name')
    }

    _params2 = {
        'param4': DataLink(identifier='param4', description='', datatype=float, obj=values, name='a')
    }

    params = {
        'params1': _params1,
        'params2': _params2
    }

    params['params1']['param1'].set(8.2)
    a = params['params1']['param1'].get()
    print(a)


if __name__ == '__main__':
    main()
