class Logger:
    @staticmethod
    def info(msg):
        print(f"\033[1;32m[INFO]\033[0m {msg}")

    @staticmethod
    def step(msg):
        print(f"\033[1;34m[STEP]\033[0m {msg}")

    @staticmethod
    def error(msg, exc=None):
        error_msg = f"\033[1;31m[ERROR]\033[0m {msg}"
        if exc:
            error_msg += f"\n\033[1;31m[EXCEPTION]\033[0m {type(exc).__name__}: {str(exc)}"
        print(error_msg)
        if exc:
            raise exc from None  # Re-raise with original traceback

    @staticmethod
    def bench(stage, t0, t1):
        Logger.info(f"{stage} completed in {t1 - t0:.3f} seconds")

    @staticmethod
    def macro(msg):
        print(f"\033[1;33m[{'█' * 10}]\033[0m {msg}")

    @staticmethod
    def micro(msg):
        print(f"\033[1;35m    [{'█' * 10}]\033[0m {msg}")