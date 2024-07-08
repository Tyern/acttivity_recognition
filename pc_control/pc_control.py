import subprocess

exp_pc_list = ["133.15.49.85 ", "133.15.49.86 ", "133.15.49.88 ", "133.15.49.77 "]
def get_PC():
    ip = subprocess.check_output(['hostname', '-s', '-I']).decode('utf-8')[:-1]
    if ip in exp_pc_list:
        return exp_pc_list.index(ip)
    else:
        return None