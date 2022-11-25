import subprocess
import os

# os.system(" > pt.txt")
# os.system(" > omp.txt")
os.system(' > luomp.txt')

for i in range(16, 0, -1):
    cmd = "./luomp 8000 " + str(i) + " >> luomp.txt"
    print(cmd)
    os.system(cmd)


