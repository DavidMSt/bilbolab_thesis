# core/providers.py
from typing import Protocol, Any, Optional

# ----- Contracts (what Core/others rely on) -----
class LoggingProvider(Protocol):
    @property
    def tick(self) -> int: ...
    def getData(self, signals, start_index: int, end_index: int) -> Any: ...
    @property
    def num_samples(self) -> int: ...


class MainProvider(Protocol):
    def getSample(self) -> Any: ...

# ----- Registry (who is currently providing) -----
_logging: Optional[LoggingProvider] = None
_main: Optional[MainProvider] = None

def set_logging_provider(p: LoggingProvider) -> None:
    global _logging; _logging = p

def get_logging_provider() -> LoggingProvider:
    if _logging is None: raise RuntimeError("Logging provider not registered")
    return _logging


def set_main_provider(p: MainProvider) -> None:
    global _main; _main = p

def get_main_provider() -> MainProvider:
    if _main is None: raise RuntimeError("Main provider not registered")
    return _main
