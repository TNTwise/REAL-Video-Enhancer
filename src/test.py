import subprocess
import threading

def print_output(pipe):
    while True:
        line = pipe.readline()
        if not line:
            break
        print(line, end="")

def run_subprocess_with_realtime_output(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        bufsize=1,  # Line-buffered output
        universal_newlines=True  # Ensure newline translation
    )

    stdout_thread = threading.Thread(target=print_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr,))

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to finish
    process.wait()

    # Wait for the output threads to finish printing
    stdout_thread.join()
    stderr_thread.join()

    return process.returncode

def main():
    command = "echo h && sleep 5"
    returncode = run_subprocess_with_realtime_output(command)
    print(f"Command finished with return code: {returncode}")

if __name__ == "__main__":
    main()