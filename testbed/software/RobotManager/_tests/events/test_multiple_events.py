import time

from _tests.events.events import Event
from core.utils.time import setTimeout

if __name__ == '__main__':
    event1 = Event()


    def fire_event():
        event1.set(data="HALLO")
        time.sleep(0.00000001)
        event1.set(data="YOU")


    def event_callback(data):
        print(data)


    event1.on(event_callback)

    setTimeout(fire_event, 1)

    while True:
        time.sleep(1)
