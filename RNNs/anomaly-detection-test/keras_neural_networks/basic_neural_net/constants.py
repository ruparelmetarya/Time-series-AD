file = "../../../../../Desktop/CpuPerc.cpu.csv"

"""Reading files into pandas dataframes"""
COLUMNS = ["time_stamp", "cpu_user", "cpu_system", "cpu_wait", "memory_cached", "memory_free", "memory_used",
           "load_longterm", "load_midterm", "load_short_term", "cpu_user_normal", "cpu_system_normal",
           "cpu_wait_normal", "memory_cached_normal", "memory_free_normal", "memory_used_normal",
           "load_longterm_normal", "load_midterm_normal"]

hbase_file =  "../../data/na44_data.csv"

hbase_columns = [
    'time_stamp','cpu_user', 'cpu_system', 'cpu_idle', 'cpu_wait', 'cpu_steal', 'cpu_nice',
    'system:interrupt', 'system:softirq', 'system:shortterm', 'system:midterm', 'system:longterm',
    'memory_used', 'memory_free', 'memory_cached', 'system:buffered', 'system:slab_recl', 'system:slab_unrecl'
]

