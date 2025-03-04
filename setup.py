import paramiko
import time
import shlex

def query(message):
    hostname = "longleaf.unc.edu"
    port = 22
    username = ""
    password = "" 
    job_check_command = "squeue --me | grep interact"
    start_job_command = "salloc -t 1:00:00 -p volta-gpu --mem=10g -N 1 -n 1 --qos gpu_access --gpus=1  --job-name=interactive_gpu"

    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, port=port, username=username, password=password)
        print(f"Connected to {hostname}")
        stdin, stdout, stderr = ssh_client.exec_command(job_check_command)
        job_status = stdout.read().decode().strip()

        if not job_status:
            print("No interactive GPU session found. Starting a new one...")
            ssh_client.exec_command(start_job_command)

        node_name = None
        while not node_name:
            stdin, stdout, stderr = ssh_client.exec_command("squeue --me --format='%N' | tail -n 1")
            node_name = stdout.read().decode().strip()
            if not node_name:
                print("Waiting for node assignment...")
                time.sleep(5)
                
        print(f"Allocated Node: {node_name}")
        ssh_shell = ssh_client.invoke_shell()
        ssh_shell.send(f"ssh {node_name}\n")
        print(f"Connected to GPU node: {node_name}")
        
        safe_message = shlex.quote(message)
        execution_commands = [
            "cd /work/users/c/h/chidaksh/hackathons/hack2",
            "module load anaconda",
            "source ~/.bashrc",
            "conda activate hackathon",
            f"python planner_inference.py {safe_message}"
        ]
        
        for cmd in execution_commands:
            print(f"running cmd: {cmd}")
            ssh_shell.send(cmd + "\n")
            wait_for_prompt(ssh_shell)

        output = read_output(ssh_shell)
        print("Execution Output:\n", output)

        ssh_shell.close()
        ssh_client.close()
        print("Connection closed.")

        return output if output else None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def wait_for_prompt(ssh_shell, timeout=600):
    buffer = ""
    start_time = time.time()
    while True:
        if ssh_shell.recv_ready():
            chunk = ssh_shell.recv(1024).decode()
            buffer += chunk
            print(chunk, end="")

        if buffer.endswith("$ ") or buffer.endswith("# "):
            break

        if time.time() - start_time > timeout:
            print("Timeout waiting for command to finish.")
            break

def read_output(ssh_shell):
    buffer = ""
    while ssh_shell.recv_ready():
        chunk = ssh_shell.recv(1024).decode()
        buffer += chunk
        print(chunk, end="")
    return buffer

message = "How many p's are in the string 'rappirappu'?"
query(message)
