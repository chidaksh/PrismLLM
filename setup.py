import paramiko

hostname = "longleaf.unc.edu"
port = 22
username = ""
password = ""

commands = [
    "cd /work/users/c/h/chidaksh/hackathons/hack1 && module load anaconda && python example.py"
]

try:
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, port=port, username=username, password=password)

    print(f"Connected to {hostname}")

    for command in commands:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(f"Executing: {command}")
        print("Output:")
        print(stdout.read().decode())
        print("Errors:")
        print(stderr.read().decode())

    ssh_client.close()
    print("Connection closed.")

except Exception as e:
    print(f"An error occurred: {e}")
