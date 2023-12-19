import os
import importlib
import inspect

# Get the current folder
current_folder = os.getcwd()

# Iterate over all files in the current folder
for file in os.listdir(current_folder):
    # Check if the file is a Python module
    if file.endswith(".py"):
        # Get the module name
        module_name = os.path.splitext(file)[0]
        
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Get all classes in the module
        classes = inspect.getmembers(module, inspect.isclass)
        
        # Print the class names
        for class_name, _ in classes:
            print(class_name)
