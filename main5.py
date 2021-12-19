from multiprocessing import Process


class test():
    def __init__(self, m):
        print("Start")


        self.start(m)

    def start(self,m):
        self.p = Process(target=self.f, args=(m,))
        self.p.start()

    def f(self, m):
        self.m = m
        i = 2
        while True:
            import time

            time.sleep(1)
            print('hello', self.m)
            i += 1
            self.m.append(i)


if __name__ == '__main__':
    m = []
    m.append(1)
    testy = test(m)
    import time

    time.sleep(5)
    while True:
        print("--->", m)

    testy.p.terminate()
