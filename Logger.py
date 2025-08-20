import os
from enum import IntEnum

class LogLevel(IntEnum):
    """Enum for logging levels, ordered by verbosity"""
    NONE = 0   # No logging
    WARNING = 1  # No logging
    ERROR = 2  # Only errors
    STEP = 3  # Errors and steps
    MACRO = 4  # Errors, steps, and macros
    INFO = 5   # Errors, steps, macros, and info
    MICRO = 6  # All messages, including micro

class Logger:
    # Default log level (can be overridden by LOG_LEVEL environment variable)
    _log_level = LogLevel.MACRO

    @classmethod
    def set_log_level(cls, level):
        """Set the logging level (string or LogLevel)"""
        if isinstance(level, str):
            level = level.upper()
            if level not in LogLevel.__members__:
                raise ValueError(f"Invalid log level: {level}. Choose from {list(LogLevel.__members__)}")
            level = LogLevel[level]
        cls._log_level = level
        print(f"\033[1;36m[LOGGER]\033[0m Log level set to {level.name}")

    @classmethod
    def __init__(cls):
        """Initialize log level from environment variable LOG_LEVEL"""
        env_level = os.getenv('LOG_LEVEL', 'MACRO').upper()
        try:
            cls.set_log_level(env_level)
        except ValueError as e:
            print(f"\033[1;31m[ERROR]\033[0m {e}. Using default log level MACRO")
            cls._log_level = LogLevel.MACRO

    @staticmethod
    def info(msg):
        """Log INFO message if level allows"""
        if Logger._log_level >= LogLevel.INFO:
            print(f"\033[1;32m[INFO]\033[0m {msg}")

    @staticmethod
    def step(msg):
        """Log STEP message if level allows"""
        if Logger._log_level >= LogLevel.STEP:
            print(f"\033[1;34m[STEP]\033[0m {msg}")

    @staticmethod
    def error(msg, exc=None):
        """Log ERROR message if level allows and optionally raise exception"""
        if Logger._log_level >= LogLevel.ERROR:
            error_msg = f"\033[1;31m[ERROR]\033[0m {msg}"
            if exc:
                error_msg += f"\n\033[1;31m[EXCEPTION]\033[0m {type(exc).__name__}: {str(exc)}"
            print(error_msg)
        if exc:
            raise exc from None

    @staticmethod
    def bench(stage, t0, t1):
        """Log benchmark timing if level allows"""
        if Logger._log_level >= LogLevel.INFO:
            Logger.info(f"{stage} completed in {t1 - t0:.3f} seconds")

    @staticmethod
    def macro(msg):
        """Log MACRO message if level allows"""
        if Logger._log_level >= LogLevel.MACRO:
            print(f"\033[1;33m[{'█' * 10}]\033[0m {msg}")

    @staticmethod
    def micro(msg):
        """Log MICRO message if level allows"""
        if Logger._log_level >= LogLevel.MICRO:
            print(f"\033[1;35m    [{'█' * 10}]\033[0m {msg}")

    def warning(msg):
        """Log WARNING message if level allows"""
        if Logger._log_level >= LogLevel.WARNING:
            print(f"\033[1;35m    [{'█' * 10}]\033[0m {msg}")