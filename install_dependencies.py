import subprocess

def install_dependencies(file_path):
    command = f'pip install -r {file_path}'
    subprocess.run(command, shell=True)

# Specify the file path of the requirements file
requirements_file = 'requirements.txt'

# Install the dependencies
install_dependencies(requirements_file)
