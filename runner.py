"""Tiny job queue: python runner.py jobs.txt NWORKERS. Each line: '<done_marker_path> ::: <shell command>'.
Jobs whose marker exists are skipped; workers claim jobs through atomic lock files."""
import sys, os, subprocess, time
from multiprocessing import Process

def worker(wid, path):
    while True:
        jobs = [l.rstrip('\n').split(' ::: ', 1) for l in open(path) if ' ::: ' in l and not l.startswith('#')]
        todo = None
        for marker, cmd in jobs:
            if os.path.exists(marker): continue
            lock = marker + '.lock'
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); todo = (marker, cmd, lock); break
            except FileExistsError:
                continue
        if todo is None: return
        marker, cmd, lock = todo; t = time.time()
        r = subprocess.run(cmd, shell=True, cwd='.')
        with open('./logs/runner.log', 'a') as f:
            f.write(f'{time.strftime("%H:%M:%S")} w{wid} rc={r.returncode} {time.time() - t:.0f}s {marker}\n')
        if r.returncode != 0: open(marker + '.failed', 'w').write(cmd)
        os.remove(lock)
        if r.returncode != 0: open(marker, 'w').write('FAILED')

if __name__ == '__main__':
    ps = [Process(target=worker, args=(i, sys.argv[1])) for i in range(int(sys.argv[2]))]
    for p in ps: p.start()
    for p in ps: p.join()
