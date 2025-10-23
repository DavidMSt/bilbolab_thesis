import logging


def disableAllOtherLoggers(module_name=None):
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name != module_name:
            log_obj.disabled = True

def disableLoggers(loggers: list):
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name in loggers:
            log_obj.disables = True

def setLoggersLevel(loggers, level):
    ...