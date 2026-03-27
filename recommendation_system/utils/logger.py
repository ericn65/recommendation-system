import logging


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with the specified name.

    This function centralizes the logging configuration, ensuring that all modules
    in the project use a consistent logging format and handlers. It sets up a
    console handler with a logging level and format specified in the project's
    configuration. If additional handlers (e.g., file handlers) are required,
    they can be added here.
    Change the logging level if wanted in this code.

    Parameters
    ----------
    name : str
        The name of the logger, typically the `__name__` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(
            logging.DEBUG
        )  # Change logger level here if wanted (another change needed bellow)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Change logger level here if wanted
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    return logger
