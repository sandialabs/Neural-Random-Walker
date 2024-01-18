# Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
# certain rights in this software.
# Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
import sys
import logging
import time
import os

# Check if nxsdk is available for running on loihi
try:
    from nxsdk.logutils.nxlogging import get_logger, NxSDKLogger, timedContextLogging, installColoredLogs
    useNxSDK = True
except ImportError as e:
    useNxSDK = False

try:
    import coloredlogs
except ImportError:
    pass

rw_logger = logging.getLogger("NRW")
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    bold_green = "\x1b[32;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)-4s:%(name)s %(relativeCreated)12d ms - %(message)s"

    FORMATS = {
        logging.DEBUG: bold_green + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,datefmt="%m/%d/%Y %I:%M:%S %p")
        return formatter.format(record)

def setup_logging(log_file='log.file',useLogFile=False,log_level=logging.DEBUG):
    os.environ["COLOREDLOGS_DATE_FORMAT"] = "%m/%d/%Y %I:%M:%S %p"
    os.environ["COLOREDLOGS_LOG_FORMAT"] = "%(asctime)s %(levelname)-4s:%(name)s %(relativeCreated)12d ms - %(message)s"
    logging.root.setLevel(log_level)


    # Set up logging.
    if useLogFile:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(CustomFormatter())
        file_handler.setLevel(log_level)
        rw_logger.addHandler(file_handler)

    print_handler = logging.StreamHandler(sys.stdout)
    print_handler.setFormatter(CustomFormatter())
    print_handler.setLevel(log_level)
    rw_logger.addHandler(print_handler)
    rw_logger.propagate = False
    coloredlogs.install(level=log_level,logger=rw_logger,formatter=print_handler.formatter,stream=sys.stdout,isatty=True)

    # These lines suppress the inherent logging done by nxsdk from propagating to my logger
    if useNxSDK:
        nxsdk_logger = get_logger() # nxsdk function that returns the global nxsdk logger
        nxsdk_logger.setLevel(log_level)
        nxsdk_logger.propagate = False
        nxsdk_logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s %(levelname)-4s:%(name)s %(relativeCreated)12d ms - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"))
        installColoredLogs(level=log_level,logger=nxsdk_logger,formatter=nxsdk_logger.handlers[0].formatter)