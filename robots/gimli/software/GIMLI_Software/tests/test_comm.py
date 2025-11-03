#!/usr/bin/env python3
"""
UGV02 Robot Controller for Raspberry Pi
=======================================

This module provides a Python interface to control and monitor
the Waveshare UGV02 robot via UART (e.g. /dev/ttyAMA0).

Features:
- Motor speed control
- OLED text control
- Continuous state feedback (position, IMU)
- Threaded serial reader
- Clean shutdown via register_exit_callback

Protocol reference:
https://www.waveshare.com/wiki/UGV02#Host_Computer_Usage_Tutorial
"""



from core.utils.exit import register_exit_callback

# ======================================================================
# === CONFIGURATION ====================================================
# ======================================================================



# ======================================================================
# === DATA STRUCTURES ==================================================
# ======================================================================



# ======================================================================
# === MAIN CLASS =======================================================
# ======================================================================




# ======================================================================
# === EXAMPLE USAGE ====================================================
# ======================================================================

def main():
    """Example usage: move in a circle for a few seconds, then stop."""
    ugv = UGV02()

    try:
        start_time = time.time()
        while time.time() - start_time < 10:
            ugv.set_motor_speed(0.2, -0.2)
            data = ugv.get_data()
            print(f"L={data.speed_left:.2f}, R={data.speed_right:.2f}, roll={data.roll:.2f}, pitch={data.pitch:.2f}")
            time.sleep(HEARTBEAT_INTERVAL)
        ugv.stop()
    finally:
        ugv.close()


if __name__ == "__main__":
    main()
