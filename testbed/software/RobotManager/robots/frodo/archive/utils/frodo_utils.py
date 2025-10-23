import time
from robots.frodo.archive.frodo import Frodo


# === TEST RESPONSE TIME ===============================================================================================
def test_response_time(frodo: Frodo, iterations=10, print_response_time=False):
    """
    Measures the response time of the Frodo robot's test method over multiple iterations.

    Args:
        frodo (Frodo): The Frodo robot instance.
        iterations (int, optional): Number of test iterations. Defaults to 10.
        print_response_time (bool, optional): Whether to print individual response times. Defaults to False.

    Logs:
        - Total number of timeouts.
        - Maximum, minimum, and average response times in milliseconds.
    """
    response_times: list[(None, float)] = [None] * iterations  # List to store response times
    timeouts = 0  # Counter for timeouts

    frodo.logger.info("Testing response time")

    # Perform an initial write to check if the robot responds
    data = frodo.test("HALLO", timeout=1)

    if data is None:
        frodo.logger.warning("Initial write timed out")
        return  # Exit the function if initial test fails

    for i in range(iterations):
        start = time.perf_counter()  # Record start time
        data = frodo.test("HALLO", timeout=1)  # Send test message

        if data is None:
            timeouts += 1  # Increment timeout counter
            response_times[i] = None
        else:
            response_times[i] = time.perf_counter() - start  # Calculate response time

        # Log response time or timeout occurrence
        if print_response_time and data is not None:
            frodo.logger.info(f"{i + 1}/{iterations} Response time: {(response_times[i] * 1000):.2f} ms")
        else:
            frodo.logger.warning(f"{i + 1}/{iterations} Timeout")

        time.sleep(0.25)  # Delay before next test iteration

    # Filter out None values (timeouts) from response times
    valid_times = [response_time for response_time in response_times if response_time is not None]

    # Calculate statistics
    max_time = max(valid_times)  # Maximum response time
    min_time = min(valid_times)  # Minimum response time
    avg_time = sum(valid_times) / len(valid_times)  # Average response time

    # Log results
    frodo.logger.info(f"Timeouts: {timeouts}")
    frodo.logger.info(f"Max time: {max_time * 1000:.2f} ms")
    frodo.logger.info(f"Min time: {min_time * 1000:.2f} ms")
    frodo.logger.info(f"Average time: {avg_time * 1000:.2f} ms")
