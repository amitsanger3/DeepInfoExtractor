# Import Dependencies
import logging, os
from logging.handlers import TimedRotatingFileHandler


class Logger(object):
    """
    Logs of the errors while running the script
    """

    def __init__(self, job):
        """
        Initialize
        :param job: str
                Name of the job whose logs you want to save.
        """
        self.job = job
        self.log_obj = logging.getLogger(self.job)
        self.log_obj.setLevel(logging.DEBUG)
        # logs file dir
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        self.handler = TimedRotatingFileHandler(filename='./logs/{}.log'.format(self.job),
                                                when='D', interval=1, backupCount=90,
                                                encoding='utf-8', delay=False)
        self.handler.setLevel(logging.ERROR)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.log_obj.addHandler(self.handler)
