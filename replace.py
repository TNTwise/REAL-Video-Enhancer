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
        
        if 'os.path.exists' in line:
            pattern = r'\((.*?)\)'

            matches = re.findall(pattern, line)
    
            for i in matches:
                i = str(i)
                
                
                i = i.replace("/",'",f"')
            line = line.replace(f'os.path.exists({matches[0]})',f'os.path.exists(os.path.join({i}))')

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
replace_linux_paths(sys.argv[1])

