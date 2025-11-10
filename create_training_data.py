import os, random

number_files = 1000

random_numbers = tuple([random.randint(1, 20)/10 for i in range(number_files)])


for i in range(number_files):
    os.system(f'python save_created_ask.py --outfile Testing_data3/{"{:06d}".format(i)}.complex --bits-outfile Testing_data3/{"{:06d}".format(i)}_text.txt --noise {random_numbers[i]}')
    if i % (number_files // 100) == 0:
        print(int((i / number_files)*100))

