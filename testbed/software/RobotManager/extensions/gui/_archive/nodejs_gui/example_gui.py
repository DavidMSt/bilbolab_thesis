import time

from gui.nodejs_gui.nodejs_gui import NodeJSGui


def main():
    gui = NodeJSGui()
    gui.init()
    gui.start()

    while True:
        gui.print("HALLO")
        time.sleep(1)


if __name__ == '__main__':
    main()
