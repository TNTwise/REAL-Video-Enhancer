import csv
import os
class Module:
    def __init__(self):
        self.module = {}
        for i in os.listdir():
            if 'module' in i:
                with open (f'{i}') as f:
                    f = csv.reader(f)
                    for row in f:
                        iteration=0
                        for i in row:
                            options=""
                            if iteration != 0:
                                options+=(i)
                            iteration = 1
                        self.module[row[0]] = options
        print(self.module)
    def run_AI(self):
        pass
Module()