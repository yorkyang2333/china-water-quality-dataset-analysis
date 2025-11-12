'''
A generic script to execute all other Python scripts (.py) in its directory.

To use:
1. Place this script in the directory containing the Python scripts you want to run.
2. Execute this script from the command line: python run_all.py

It will automatically find and run every other .py file in that folder.
'''
import os
import subprocess
import sys

def run_scripts_in_directory():
    """
    Finds and executes all other Python scripts (.py) in the same directory
    as this script, in alphabetical order.
    """
    try:
        # Get the absolute path of the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the name of this script to avoid running it again
        this_script_name = os.path.basename(__file__)
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g., interactive interpreter)
        script_dir = os.getcwd()
        this_script_name = None
        print("Warning: __file__ not defined. Running in current working directory.")

    print(f"--- Starting batch execution in: {script_dir} ---")
    print("-" * 50)

    # Get a sorted list of files to ensure a consistent execution order
    try:
        files_in_dir = sorted(os.listdir(script_dir))
    except FileNotFoundError:
        print(f"Error: Directory not found at {script_dir}")
        return

    executed_count = 0
    for filename in files_in_dir:
        # Check if the file is a Python script and not this script itself
        if filename.endswith(".py") and (filename != this_script_name):
            script_path = os.path.join(script_dir, filename)
            print(f"-> Running: {filename}")
            try:
                # Execute the script using the same Python interpreter that is running this script.
                # Using check=True will raise a CalledProcessError if the script returns a non-zero exit code.
                subprocess.run([sys.executable, script_path], check=True)
                print(f"   Status: Success")
                executed_count += 1
            except subprocess.CalledProcessError as e:
                print(f"   !!! Error: {filename} returned non-zero exit code {e.returncode}. !!!")
            except Exception as e:
                print(f"   !!! An unexpected error occurred while running {filename}: {e} !!!")
            finally:
                print("-" * 50)

    if executed_count == 0:
        print("No other Python scripts were found to execute.")
    else:
        print(f"--- Batch execution finished. Total scripts run: {executed_count} ---")

if __name__ == "__main__":
    run_scripts_in_directory()
