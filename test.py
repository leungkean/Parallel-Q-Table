from multiprocessing import Process, Lock
count = 0

def f(l): 
    global count
    l.acquire()
    count += 1
    print('hello world', count)
    l.release()

def main():
    lock = Lock()
    processes = []

    for num in range(8):
        processes.append(Process(target=f, args=(lock,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
