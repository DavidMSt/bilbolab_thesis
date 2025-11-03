import time

from core.utils.websockets import WebsocketClient

if __name__ == '__main__':
    websocket_client = WebsocketClient(address='gimli.lan', port=8080)
    websocket_client.connect()


    while not websocket_client.connected:
        time.sleep(1)

    websocket_client.send({
        'type': 'set_motor_speed',
        'data': {
            'left': 0.25,
            'right': 0.25
        }
    })

    time.sleep(3)
    websocket_client.send({
        'type': 'set_motor_speed',
        'data': {
            'left': 0,
            'right': 0
        }
    })

    time.sleep(5)
    print('Connected')