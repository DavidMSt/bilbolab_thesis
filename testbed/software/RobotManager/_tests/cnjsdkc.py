import dataclasses


@dataclasses.dataclass
class Test:
    a: int = 0
    b: str = ''


if __name__ == '__main__':
    config = {
        'a': 1,
        'b': 'test',
        'c': 1.0,
    }

    test = Test(**config)
    print(test)
