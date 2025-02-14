import os
import subprocess

def pidstat_cpu_mem(pid, interval=1, count=2):
    """
    Returns (cpu_usage, mem_usage) for the given PID using pidstat.
    pidstat -r -u -p <PID> 1 1 runs once, collects 1 second interval, 1 sample.
    """
    # 1) Run pidstat for CPU + Memory usage, 1-second interval, 1 sample
    # cmd = f"pidstat -r -u -p {pid} 1 1"
    cmd = f"pidstat -r -u -p {pid} {interval} {count}"
    output = subprocess.check_output(cmd, shell=True, text=True).strip()
    
    # 2) Prepare variables for final results
    cpu_usage = None
    mem_usage = None
    
    # 3) Parse each line
    for line in output.splitlines():
        line = line.strip()
        
        # Skip header lines, average lines, blank lines
        if (not line or
            line.startswith("Linux") or
            line.startswith("Average") or
            line.startswith("#") or
            "Command" in line):
            continue
        cols = line.split()
    
        if str(pid) not in cols:
            continue
        
        pid_index = cols.index(str(pid))
        # For CPU line, after PID we expect: %usr, %system, %guest, %wait, %CPU, CPU, Command
        # For MEM line, after PID we expect: minflt/s, majflt/s, VSZ, RSS, %MEM, Command
        
        # Let's count how many columns after the PID:
        after_pid = len(cols) - (pid_index + 1)

        if after_pid == 7:

            cpu_usage_str = cols[pid_index + 5]
            try:
                cpu_usage = float(cpu_usage_str)
            except ValueError:
                cpu_usage = None
        
        # MEM LINE (6 columns after PID)
        # e.g. [ minflt/s, majflt/s, VSZ, RSS, %MEM, Command ]
        elif after_pid == 6:

            mem_usage_str = cols[pid_index + 5]
            try:
                mem_usage = float(mem_usage_str)
            except ValueError:
                mem_usage = None
    
    return cpu_usage, mem_usage


