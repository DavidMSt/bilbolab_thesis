from core.utils.events import EventFlag, Event, pred_flag_equals, wait_for_events, OR
from core.utils.time import setTimeout

if __name__ == '__main__':
    event1 = Event(flags=EventFlag('trajectory_id', str))
    event2 = Event()


    def set_event():
        event1.set(data='test', flags={'trajectory_id': '123'})


    setTimeout(set_event, 2)

    data, result = wait_for_events(
        events=OR((event1, pred_flag_equals('trajectory_id', '123')), event2),
        timeout=10,
    )
    print(data)
    print(result)
