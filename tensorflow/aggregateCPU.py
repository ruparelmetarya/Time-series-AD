import csv
import pandas as pd

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

data2 = pd.read_csv('DemoCPU.csv',engine='python')
print data2['Timestamp']
timestamp = []
cpu_data = []
for i in range(len(data2['Timestamp'])) :
    timestamp.append(data2['Timestamp'][i])
    cpu_data.append(data2['system.PHX.SP1.na44:CpuPerc.cpu.user{podtype=Pri,hostState=unknown}'][i])
print timestamp
print cpu_data
temp = 0
aggregatedData = []
aggregatedTime = []
for i in my_range(0,636,10) :
    aggregatedTime.append(timestamp[i])
    count = 0
    for j in range(temp,temp+10):
        temp = j+1
        count = count + cpu_data[j]
    aggregatedData.append(count)
print len(aggregatedData)
print len(aggregatedTime)

with open('aggregatedCPU.csv', 'w') as csvfile:
    fieldnames = ['Timestamp' , 'system.PHX.SP1.na44:CpuPerc.cpu.user{podtype=Pri,hostState=unknown}']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(aggregatedTime)):
        writer.writerow({'Timestamp' : aggregatedTime[i] , 'system.PHX.SP1.na44:CpuPerc.cpu.user{podtype=Pri,hostState=unknown}' : aggregatedData[i]})
