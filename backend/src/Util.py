import os
import warnings

cwd = os.getcwd()




def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def printAndLog(message: str):
    print(message)
    log(message=message)


def log(message: str):
    with open(os.path.join(cwd, "log.txt"), "a") as f:
        f.write(message + "\n")


def currentDirectory():
    return cwd
