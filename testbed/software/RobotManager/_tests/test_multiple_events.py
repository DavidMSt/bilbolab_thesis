import time

from core.utils.events import Event, Subscriber
from core.utils.time import setTimeout

if __name__ == '__main__':
    event1 = Event()


    def fire_event():
        event1.set(data="HALLO")
        event1.set(data="YOU")


    def event_callback(data):
        print(data)


    # event1.on(event_callback)
    subscriber = Subscriber(events=event1)
    subscriber.callbacks.finished.register(event_callback)

    setTimeout(fire_event, 1)

    while True:
        time.sleep(1)
