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


#mapping metrics to file paths

file_names_map = {'cpu_system': 'cpu_system/CpuPerc.cpu.system.csv', 'cpu_user': 'cpu_user/CpuPerc.cpu.user.csv',
        'cpu_wait': 'cpu_wait/CpuPerc.cpu.wait.csv',
        'memory_cached': 'memory_cached/memory.memory.cached.csv',
        'memory_free': 'memory_free/memory.memory.free.csv',
        'memory_used': 'memory_used/memory.memory.used.csv',
        'load_longterm': 'load_longterm/load.load.longterm.csv',
        'load_midterm': 'load_midterm/load.load.midterm.csv',
        'load_shortterm': 'load_shortterm/load.load.shortterm.csv',
        'power_supply_1': 'power_supply_1/powerstats.gauge.PowerSupply1.csv',
        'power_supply_2': 'power_supply_2/powerstats.gauge.PowerSupply2.csv'
        }


def read_data_from_csv(file_name, field_names, data_array):
    with open(file_name, 'rb') as csvfile:
        field_names = field_names
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='|', fieldnames=field_names)
        for row in reader:
            data_array.append(row)
    return data_array


cpu_system = read_data_from_csv(file_names_map['cpu_system'], ['time_stamp', 'cpu_perc_system'], [])
cpu_user = read_data_from_csv(file_names_map['cpu_user'], ['time_stamp', 'cpu_perc_user'], [])
cpu_wait = read_data_from_csv(file_names_map['cpu_wait'], ['time_stamp', 'cpu_perc_wait'], [])

memory_cached = read_data_from_csv(file_names_map['memory_cached'], ['time_stamp', 'memory_cached'], [])
memory_free = read_data_from_csv(file_names_map['memory_free'], ['time_stamp', 'memory_free'], [])
memory_used = read_data_from_csv(file_names_map['memory_used'], ['time_stamp', 'memory_used'], [])


def rescale_time_stamp(rows, start, end):

    first_row = rows[start]
    first_time_stamp = first_row['time_stamp']

    for i in range(start, end):
        row = rows[i]
        rows[i] = subtract_time_stamp_by_val(row, int(first_time_stamp))
        rows[i] = divide_time_stamp_by_1000(rows[i])

    return rows

def divide_time_stamp_by_1000(row):
    timestamp = row['time_stamp']
    timestamp = timestamp/1000
    row['time_stamp'] = timestamp
    return row

def subtract_time_stamp_by_val(row, val):
    time_stamp = row['time_stamp']
    time_stamp = int(time_stamp) - val
    row['time_stamp'] = time_stamp
    return row

def get_mean_val(rows, start, end, metric_field_name):
    y = []
    for i in range(start, end):
        row = rows[i]
        y.append(float(row[metric_field_name]))
    avg = sum(y) / float(len(y))
    return avg

numInserts = 0

def fill_missing_data(rows, start, end, unit, metric_field_name):
    print 'size before filling in '
    print len(rows)
    mean = get_mean_val(rows, start, end, metric_field_name)
    for i in range(start, end-1):
        j = i + 1
        prev_timestamp = int(rows[i]['time_stamp'])
        next_timestamp = int(rows[j]['time_stamp'])
        diff = next_timestamp - prev_timestamp
        nextRow = rows[j]
        if(diff>unit):
            num = diff/unit
            for k in range(0, num-1):
                global numInserts
                numInserts = numInserts+1
                newRow = {}
                newRow['time_stamp'] = prev_timestamp + unit
                newRow[metric_field_name] = mean
                rows.insert(j+k, newRow)
    print 'size after filling in '
    print len(rows)
    return rows


cpu_system = rescale_time_stamp(cpu_system, 1, len(cpu_system))


print 'cpu_len_initial  ' + str(len(cpu_system))



memory_used = rescale_time_stamp(memory_used, 1, len(memory_used))
print 'memory_used_len_initial   '+str(len(memory_used))




#
# for i in range(1, len(cpu_system)):
#     if(memory_used[i]['time_stamp']!=cpu_system[i]['time_stamp']):
#         print 'not equal  '
#         print 'i  '+str(i)
#         print 'memory-- time_stamp  '+ str(memory_used[i]['time_stamp'])
#         print 'cpu-timestamp  ' + str(cpu_system[i]['time_stamp'])


count = 0
for i in range(1, len(cpu_system)-1):
    j = i+1
    prev = int(cpu_system[i]['time_stamp'])
    nex = int(cpu_system[j]['time_stamp'])
    if((nex-prev)!=60):
        count = count +1
print count



cpu_system = fill_missing_data(cpu_system, 1, len(cpu_system), 60, 'cpu_perc_system')
print 'cpu len_final  ' + str(len(cpu_system))
print 'numInsertsss>>>:'
print numInserts

count = 0;
for i in range(1, len(cpu_system)-1):
    j = i+1
    prev = int(cpu_system[i]['time_stamp'])
    nex = int(cpu_system[j]['time_stamp'])
    if(int(cpu_system[j]['time_stamp'])-int(cpu_system[i]['time_stamp'])>60):
        count = count +1
print count



memory_used = fill_missing_data(memory_used, 1, len(memory_used), 60, 'memory_used')
print 'memory_used_len_final   '+str(len(memory_used))

k = 0
for i in range(1, 150000):
    one = memory_used[i]['time_stamp']
    two = cpu_system[i]['time_stamp']
    if(memory_used[i]['time_stamp']!=cpu_system[i]['time_stamp']):
        k = k+1
        # print 'not equal  '
        # print 'i  '+str(i)
        # print 'memory-- time_stamp  '+ str(memory_used[i]['time_stamp'])
        # print 'cpu-timestamp  ' + str(cpu_system[i]['time_stamp'])

print k
#
# cpu_user = rescale_time_stamp(cpu_user, 1, len(cpu_user), 1, 60000)
# cpu_wait = rescale_time_stamp(cpu_wait, 1, len(cpu_wait), 1, 60000)
# memory_free = rescale_time_stamp(memory_free, 1, len(memory_free), 1, 60000)
# memory_used = rescale_time_stamp(memory_used, 1, len(memory_used), 1, 60000)








