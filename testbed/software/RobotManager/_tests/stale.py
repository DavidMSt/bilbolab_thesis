import time

from core.utils.events import Event, TIMEOUT

if __name__ == '__main__':
    event = Event()

    event.set(data="hello")

    time.sleep(2)

    data, result = event.wait(stale_event_time=5, timeout=2)
    if data is TIMEOUT:
        print("timeout")
    else:
        print("data", data)
