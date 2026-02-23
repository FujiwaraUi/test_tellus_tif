import os


filepath = os.path.dirname(os.path.abspath(__file__))
print(filepath)
print(type(filepath))
PATH_OUTPUT = os.path.join(os.path.dirname(filepath), 'output')
os.makedirs(PATH_OUTPUT, exist_ok=True)
