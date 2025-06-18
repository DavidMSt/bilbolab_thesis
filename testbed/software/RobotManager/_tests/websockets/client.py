import time

from core.utils.websockets.websockets import WebsocketClient


def on_connect(*args, **kwargs):
    print("connected")


def on_message(message, *args, **kwargs):
    print(f"message received: {message}")


def main():
    client = WebsocketClient(address='127.0.0.1', port=8000)
    client.callbacks.message.register(on_message)
    client.start()

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
