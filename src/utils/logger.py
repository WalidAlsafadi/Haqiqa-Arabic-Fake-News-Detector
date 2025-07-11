import datetime

def log(msg, level="INFO"):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{level}] [{now}] {msg}")

def log_success(msg): log(msg, level="SUCCESS")
def log_fail(msg): log(msg, level="FAIL")
def log_info(msg): log(msg, level="INFO")
def log_step(msg): log(msg, level="STEP")
def log_warn(msg): log(msg, level="WARN")
