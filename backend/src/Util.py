import os
import warnings

cwd = os.getcwd()
with open(os.path.join(cwd, "log.txt"), "w") as f:
    pass


def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def printAndLog(message: str, separate=False):
    """
    Prints and logs a message to the log file
    separate, if True, activates the divider
    """
    if separate:
        message = message + "\n" + "---------------------"
    print(message)
    log(message=message)


def log(message: str):
    with open(os.path.join(cwd, "log.txt"), "a") as f:
        f.write(message + "\n")


def currentDirectory():
    return cwd
