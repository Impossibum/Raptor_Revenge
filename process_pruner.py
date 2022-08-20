import psutil
from time import sleep


def prune_loop(sleep_duration: int):
    print("Starting process pruning.")
    rl_procs = create_instance_list()
    for proc in rl_procs:
        print(f"Killing process {proc.name()}, pid: {proc.pid}.")
        proc.terminate()
        sleep(sleep_duration)


def create_instance_list():
    proc_list = []
    for proc in psutil.process_iter():
        p_name = proc.name()
        if p_name == "RocketLeague.exe":
            proc_list.append(proc)
    return proc_list


if __name__ == "__main__":
    long_delay = 60*6*60
    short_delay = 60
    immediate_start = True
    counter = 0
    print("Pruning script activated.")

    while True:
        if not immediate_start:
            print(f"sleeping for {long_delay/60} minutes")
            sleep(long_delay)
        else:
            immediate_start = False
        prune_loop(short_delay)
        counter += 1
        print(f"Restart cycle #{counter} completed.")


