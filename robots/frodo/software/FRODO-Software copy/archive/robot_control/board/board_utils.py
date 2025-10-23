from RPi import GPIO
import board.settings as board_settings


def reset_uart():
    print("RESET UART")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(board_settings.RC_GPIO_UART_RESET, GPIO.OUT)
    GPIO.output(board_settings.RC_GPIO_UART_RESET, 1)
    GPIO.output(board_settings.RC_GPIO_UART_RESET, 0)
    GPIO.cleanup()
