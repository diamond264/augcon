import subprocess

def top_cpu_mem(pid):

    cmd_top = f"top -bn1 | awk '$1 == {pid} {{print $1, $9, $10}}'"
    output_top = subprocess.check_output(cmd_top, shell=True, text=True).strip()

    if output_top:
        t_cpu, t_mem = output_top.split()[1:]
        return float(t_cpu), float(t_mem)
    else:
        return None, None