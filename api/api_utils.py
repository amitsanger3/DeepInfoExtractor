import datetime
from config import api_log_file


def write_api_logs(log_text):
    fls = open(api_log_file, 'a+')
    msg = f"{datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')} || {log_text} \n"
    fls.write(msg)
    fls.close()