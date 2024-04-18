import os
import re
import difflib
import sys
def replace_linux_paths(file_path):
    # Define the regex pattern for matching Linux paths
    replaced_content = []
    # Read the file
    with open(file_path, 'r') as f:
        content = f.readlines()

    # Replace Linux paths with os.path.join or an OS-agnostic option
    for line in content:
        
        if 'with open' in line:
            pattern = r'\((.*?)\)'

            matches = re.findall(pattern, line)
    
            for i in matches:
                i = str(i)
                
                
                i = i.replace("/",'",f"')
            line = line.replace(f'with open({matches[0]})',f'with open((os.path.join({i})))')

        replaced_content.append(line)
    with open(file_path, 'w') as f:
        f.writelines(replaced_content)
def print_differences(file_path, old_content, new_content):
    # Compute the differences between the original and modified content
    diff = difflib.unified_diff(old_content.splitlines(), new_content.splitlines(), lineterm='')

    # Print the differences
    print(f"Differences for file: {file_path}")
    print('\n'.join(diff))

# Example usage
  # Replace with the path to your file
import os

def print_files(directory):
    # Iterate over all files and directories in the current directory
    for item in os.listdir(directory):
        # Construct the full path to the item
        item_path = os.path.join(directory, item)
        # If it's a directory, recursively call print_files on it
        if os.path.isdir(item_path):
            print_files(item_path)
        # If it's a file, print its full path
        elif os.path.isfile(item_path):
            if '.py' in item_path:
                try:
                    replace_linux_paths(item_path)
                except:
                    pass

print_files(sys.argv[1])
        

