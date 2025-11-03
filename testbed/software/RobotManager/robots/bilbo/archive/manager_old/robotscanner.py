from robots.bilbo.robot.bilbo_definitions import TWIPR_IDS
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.exit import register_exit_callback
from core.utils.network.network import pingAddresses, resolveHostname
import threading
import time
from core.utils.logging_utils import Logger

logger = Logger('scanner')

print("DO I FIND THIS?")

@callback_definition
class RobotScannerCallbacks:
    found: CallbackContainer
    lost: CallbackContainer


class RobotScanner:
    callbacks: RobotScannerCallbacks

    def __init__(self, robot_ids, scan_interval=3, failure_threshold=40):
        """
        Initializes the TWIPR_Scanner class.

        :param robot_ids: List of robot hostnames to scan.
        :param scan_interval: Time interval (in seconds) between scans.
        :param failure_threshold: Time duration (in seconds) after which a robot is considered lost if it remains unreachable.
        """
        self.robot_ids = robot_ids
        self.scan_interval = scan_interval
        self.failure_threshold = failure_threshold

        self.active_robots = {}  # {hostname: ip}
        self.unreachable_since = {}  # {hostname: timestamp}
        self.scanning = False
        self._lock = threading.Lock()
        self._thread = None

        self.callbacks = RobotScannerCallbacks()

        register_exit_callback(self.stop)

    # ==================================================================================================================
    def _scan_network(self):
        while self.scanning:
            reachable_robots = pingAddresses(self.robot_ids)
            current_time = time.time()
            with self._lock:
                # Detect newly active robots
                for hostname, is_reachable in reachable_robots.items():
                    if is_reachable:
                        if hostname not in self.active_robots:
                            ip_address = resolveHostname(hostname)
                            if ip_address:
                                self.active_robots[hostname] = ip_address
                                if hostname in self.unreachable_since:
                                    del self.unreachable_since[hostname]
                                for callback in self.callbacks.found:
                                    callback(hostname, ip_address)
                    else:
                        if hostname in self.active_robots:
                            if hostname not in self.unreachable_since:
                                self.unreachable_since[hostname] = current_time

                # Detect lost robots
                for hostname in list(self.active_robots.keys()):
                    if hostname in self.unreachable_since:
                        if current_time - self.unreachable_since[hostname] >= self.failure_threshold:
                            for callback in self.callbacks.lost:
                                callback(hostname, self.active_robots[hostname])

                            del self.active_robots[hostname]
                            del self.unreachable_since[hostname]

            time.sleep(self.scan_interval)

    def start(self):
        """Starts the scanning process."""
        logger.info("Start Robot Scanner")
        if not self.scanning:
            self.scanning = True
            self._thread = threading.Thread(target=self._scan_network)
            self._thread.start()

    def stop(self, *args, **kwargs):
        """Stops the scanning process."""
        if self.scanning:
            self.scanning = False
            if self._thread:
                self._thread.join()
        logger.info("Close Robot Scanner")

    def get_active_robots(self):
        """Returns a dictionary of active robots."""
        with self._lock:
            return dict(self.active_robots)


if __name__ == '__main__':

    scanner = RobotScanner(
        robot_ids=TWIPR_IDS,
        scan_interval=5,
        failure_threshold=40  # Adjust the failure threshold as needed
    )
    scanner.start()

    while True:
        time.sleep(10)
