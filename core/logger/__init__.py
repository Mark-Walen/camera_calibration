import logging


class Logger:
    def __init__(self, name="NodeLogger", fmt='%(asctime)s [%(levelname)s]: %(message)s'):
        # Initialize the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG level to capture all messages
        ch = logging.StreamHandler()  # StreamHandler outputs logs to the console
        ch.setLevel(logging.DEBUG)  # Set the level for the handler to DEBUG
        formatter = logging.Formatter(fmt)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log(self, msg, level=logging.INFO):
        """
        Logs a message at the specified level.
        """
        if level == logging.DEBUG:
            self.logger.debug(msg)
        elif level == logging.INFO:
            self.logger.info(msg)
        elif level == logging.WARNING:
            self.logger.warning(msg)
        elif level == logging.ERROR:
            self.logger.error(msg)
        elif level == logging.CRITICAL:
            self.logger.critical(msg)

    def get_logger(self):
        """
        Returns the logger instance.
        """
        return self.logger

    def debug(self, msg):
        self.log(msg, logging.DEBUG)

    def info(self, msg):
        self.log(msg, logging.INFO)

    def warning(self, msg):
        self.log(msg, logging.WARNING)

    def error(self, msg):
        self.log(msg, logging.ERROR)

    def critical(self, msg):
        self.log(msg, logging.CRITICAL)


if __name__ == '__main__':
    import time

    logger = Logger()
    logger.info("Starting the node")
    try:
        for i in range(5):
            logger.debug(f"Processing iteration {i}")
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    logger.info("Node finished execution")
