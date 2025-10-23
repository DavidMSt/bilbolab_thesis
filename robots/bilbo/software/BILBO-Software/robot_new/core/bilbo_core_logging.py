import queue
import threading
from copy import copy

from core.utils.dataclass_utils import asdict_optimized
from core.utils.dict_utils import optimized_generate_empty_copies, copy_dict
from core.utils.exit import register_exit_callback
from core.utils.h5 import H5PyDictLogger
from core.utils.logging_utils import Logger
from core.utils.time import TimeoutTimer, PerformanceTimer
from robot.logging.bilbo_sample import BILBO_Sample
from robot_new.core.bilbo_core_common import BILBO_Core_Common
from robot_new.lowlevel.bilbo_lowlevel import BILBO_LowLevel
from robot_new.lowlevel.definitions.stm32_sample import BILBO_LL_Sample, SAMPLE_BUFFER_LL_SIZE

SAMPLE_TIMEOUT_TIME = 0.5


# === BILBO CORE LOGGING ===============================================================================================
class BILBO_Core_Logging:
    common: BILBO_Core_Common
    lowlevel: BILBO_LowLevel

    sample: BILBO_Sample

    # -- Private Variables --
    _h5Logger: H5PyDictLogger
    _sample_timeout_timer: TimeoutTimer
    _num_samples: int = 0
    _samples_queue: queue.Queue

    _lock: threading.Lock

    # -- Caches for optimized data access --
    _copy_cache_full: list
    _copy_cache_ll: list
    _out_samples: list[dict]

    _initialized: bool = False
    _firstSampleReceived: bool = False

    # === INIT =========================================================================================================
    def __init__(self, common: BILBO_Core_Common, lowlevel: BILBO_LowLevel):
        self.common = common
        self.lowlevel = lowlevel

        self.logger = Logger("BILBO_CORE_LOGGING", "DEBUG")

        # --- Tick ---
        self.tick = 0

        # --- Samples Queue ---
        self._samples_queue = queue.Queue()

        # --- H5 Logger ---
        self._h5Logger = H5PyDictLogger(filename='log.h5')
        self._lock = threading.Lock()

        # --- Sample Timeout Timer ---
        self._sample_timeout_timer = TimeoutTimer(timeout_time=SAMPLE_TIMEOUT_TIME,
                                                  timeout_callback=self._timeout_callback)

        # -- Exit Handling --
        register_exit_callback(self.close, priority=10)

    # === METHODS ======================================================================================================
    def init(self):
        self._initialize_caches()

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.lowlevel.callbacks.samples.register(self._stm32Samples_callback)

    # ------------------------------------------------------------------------------------------------------------------
    def close(self, *args, **kwargs):
        self._h5Logger.close()
        self._sample_timeout_timer.stop()
        self._sample_timeout_timer.close()
        self.logger.info('BILBO Logging closed')

    # ------------------------------------------------------------------------------------------------------------------
    def update(self) -> None:

        # 1. General things
        if not self._initialized:
            self.logger.error("BILBO_Core_Logging not initialized, but update() called.")
            return

        if not self._firstSampleReceived:
            self._firstSampleReceived = True
            self._sample_timeout_timer.start()

        # Reset timeout early
        self._sample_timeout_timer.reset()

        # 2. Collect HL sample
        high_level_sample = asdict_optimized(BILBO_Sample())  # TODO: Here I should collect samples from all subsystems
        # Drain all available batches
        batch_index = 0
        while True:
            try:
                batch = self._samples_queue.get_nowait()
                batch_index += 1
                if batch_index > 1:
                    self.logger.important(f"Working on batch {batch_index}")
            except queue.Empty:
                break

            performance_timer = PerformanceTimer(name='Update', print_output=True)

            # 3. Copy HL sample into each preallocated out-sample
            for i in range(SAMPLE_BUFFER_LL_SIZE):
                dst = self._out_samples[i]
                copy_dict(dict_from=high_level_sample, dict_to=dst, structure_cache=self._copy_cache_full)

            # 4. Patch tick and time in each of the copies + copy LL per sample
            base_tick = high_level_sample['general']['tick'] + batch_index * SAMPLE_BUFFER_LL_SIZE
            sample_time = high_level_sample['general']['sample_time_ll']

            for i in range(SAMPLE_BUFFER_LL_SIZE):
                dst = self._out_samples[i]
                ti = base_tick + i
                dst['general']['tick'] = ti
                dst['general']['time'] = ti * sample_time

                copy_dict(dict_from=batch[i],
                          dict_to=dst['lowlevel'],
                          structure_cache=self._copy_cache_ll)

            # 5. Append to the H5 file
            self._h5Logger.appendSamples(self._out_samples)

            # 6. Bookkeeping
            self._num_samples += SAMPLE_BUFFER_LL_SIZE
            elapsed_time = performance_timer.stop()

            if self._num_samples % 2000 == 0:
                self.logger.debug(f"Samples collected: {self._num_samples}")

            if elapsed_time > 0.1:
                self.logger.warning(f"Logging took {elapsed_time:.2f}s")

    # === PRIVATE METHODS ==============================================================================================
    def _initialize_caches(self) -> None:
        if self._initialized:
            return

        # Schema based on dataclasses (structure only, values None)
        hl_template = asdict_optimized(BILBO_Sample())  # includes 'lowlevel' in the dataclass
        ll_template = asdict_optimized(BILBO_LL_Sample())

        schema = dict(hl_template)
        schema['lowlevel'] = ll_template

        self._h5Logger.init(schema)
        self._h5Logger.start('w')

        # Preallocate N dicts with identical structure and None leaves
        self._out_samples = optimized_generate_empty_copies(schema, SAMPLE_BUFFER_LL_SIZE)

        # Build caches using dict_to (out_samples[0]) so shapes match destination
        self._copy_cache_full = copy_dict(dict_from=schema,
                                          dict_to=self._out_samples[0],
                                          structure_cache=None)
        self._copy_cache_ll = copy_dict(dict_from=ll_template,
                                        dict_to=self._out_samples[0]['lowlevel'],
                                        structure_cache=None)

        # OPTIONAL: build a HL-only cache that excludes 'lowlevel' paths
        # Comment these two lines out if you prefer to keep _copy_cache_full.
        # self._copy_cache_hl = [p for p in self._copy_cache_full if (p and p[0] != 'lowlevel')]

        self._initialized = True

    # ------------------------------------------------------------------------------------------------------------------
    def _timeout_callback(self):
        self.logger.error('Sample timeout')

    # ------------------------------------------------------------------------------------------------------------------
    def _stm32Samples_callback(self, samples: list[BILBO_LL_Sample]):
        self._samples_queue.put(copy(samples))

    # ------------------------------------------------------------------------------------------------------------------
