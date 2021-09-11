import tensorflow as tf
import numpy as np
import csv
import tempfile
import datetime as dt
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from numpy import median
from random import choice, shuffle
from numpy import array

COLUMNS = ["timestamp", "memory_cached"]
file_name = "memory.memory.cached.csv"
data_frame = pd.read_csv(file_name, names=COLUMNS, skipinitialspace=True, na_values=[])
data_frame.fillna('', inplace=True)

ifile = open(file_name, "rb")

reader = csv.reader(ifile)
rows = []

with open(file_name, 'rb') as csvfile:
    field_names = ['time_stamp', 'memory_cached']
    reader = csv.DictReader(csvfile, delimiter=',', quotechar='|', fieldnames=field_names)
    for row in reader:
        rows.append(row)


def check_diff(rows, total_diff_1, total_diff_2, a):
    for i in range(1, 143474):
        j = i+1
        timeStamp1 = (int)(rows[i]['time_stamp'])
        timeStamp2 = (int)(rows[j]['time_stamp'])
        #print(timeStamp2-timeStamp1)
        difference = timeStamp2 - timeStamp1
        total_diff_2+=difference
        if(difference!=60000):
            a+=1
            total_diff_1+=(difference-60000)
    print total_diff_1
    print a
    print total_diff_2


def check_smallest_diff(rows, total_diff_1, total_diff_2, a):
    for i in range(1, 143474):
        j = i+1
        timeStamp1 = (int)(rows[i]['time_stamp'])
        timeStamp2 = (int)(rows[j]['time_stamp'])
        difference = timeStamp2 - timeStamp1

        if(difference<60000):
            print 'FALSE'


def check_multiple_diff(rows, total_diff_1, total_diff_2, a):
    for i in range(1, 143474):
        j = i + 1
        timeStamp1 = (int)(rows[i]['time_stamp'])
        timeStamp2 = (int)(rows[j]['time_stamp'])
        difference = timeStamp2 - timeStamp1
        moddd = difference%60000
        if (difference%60000!=0):
            print moddd
            print 'FALSE'


def scale_timestamp_by_thousand(rows, start, end):
    for i in range(start, end):
        row = rows[i]
        time_stamp = (int)(row['time_stamp'])
        time_stamp = time_stamp / 1000
        row['time_stamp'] = time_stamp
    return rows


def rescale_time_stamp(rows, start, end, new_unit, prev_unit):
    first_row = rows[start]
    second_row = rows[start + 1]
    diff = int(first_row['time_stamp']) - int(second_row['time_stamp'])
    old_time_stamp = first_row['time_stamp']
    first_row['time_stamp'] = 0

    for i in range(start + 1, end):
        row = rows[i]
        prev_row_time_stamp = rows[i - 1]['time_stamp']
        diff = int(row['time_stamp']) - int(old_time_stamp)
        old_time_stamp = row['time_stamp']
        row['time_stamp'] = prev_row_time_stamp + (new_unit * (diff / prev_unit))
    print 'hello'
    return rows
#checkDiff(rows, 0,0,0)
#check_smallest_diff(rows, 0, 0, 0)
#check_multiple_diff(rows, 0, 0, 0)





#scale_timestamp(rows)
#print rows[1]['time_stamp']

#rescale time stamp
rows = rescale_time_stamp(rows, 1, 140000, 1, 60000)


#creating arrays for graph plotting
x = []
y = []
for i in range(1,140000):
    row = rows[i]
    x.append((row['time_stamp']))
    y.append(float(row['memory_cached']))

#plotting cpu perc vs time
# plt.xlabel('time')
# plt.ylabel('CPU-Perc-Idle')
# plt.plot(x, y)
# plt.savefig('cpu_perc_idle_VS_TIME.png')


min_memory_val = min(y)
max_memory_val = max(y)
avg_memory = sum(y)/float(len(y))
median = median(y)

print 'Some stats on Memory data from last 180 days'
print 'Minimum value:   '+ str(min_memory_val)
print 'Maximum value:   '+ str(max_memory_val)
print 'Average value:   '+ str(avg_memory)
print 'Median :         '+ str(median)

# #plotting just cpu perc
# plt.ylabel('Memory_Cached')
# plt.plot(y)
# plt.grid(True)
# plt.savefig('memory_cached.png')















# ofile = open(file_name, "wb")
#
# # fieldnames = ['timeStamp', 'cpu']
# # writer = csv.DictWriter(ofile, fieldnames=fieldnames)
# # writer.writerow({'cpu':'hello'})
#
#
# ifile.close()
# ofile.close()


# df = pd.DataFrame(columns=("Time", "Sales"))
# start_date = dt.datetime(2015, 7,1)
# end_date = dt.datetime(2015, 7,10)
# daterange = pd.date_range(start_date, end_date)
#
#
#
# for single_date in daterange:
#  row = dict(zip(["Time", "Sales"],
#      [single_date,
#      int(50*np.random.rand(1))]))
#  row_s = pd.Series(row)
#  row_s.name = single_date.strftime("%b %d")
#  df = df.append(row_s)
#
#
# df.ix["Jul 01":"Jul 07", ["Time", "Sales"]].plot()
#
#
# plt.ylim(0, 50)
# plt.xlabel("Sales Date")
# plt.ylabel("Sale Value")
# plt.title("Plotting Time")
# plt.show()



