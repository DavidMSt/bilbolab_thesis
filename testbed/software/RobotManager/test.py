from __future__ import annotations


class A:
    def __init__(self):
        self.x = 3

    def test(self, input: B):
        print(input.y)

    class B:
        def __init__(self):
            self.y = 4


