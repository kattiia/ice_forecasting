import subprocess

def run_script(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    print("🔹 Start to process the data...")
    run_script("process_data.py")

    print("🔹 Load data in database...")
    run_script("load_to_db.py")

    print("Process is finished!")
