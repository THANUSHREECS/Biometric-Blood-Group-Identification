import os

def unignore_results():
    if not os.path.exists('.gitignore'):
        return
    with open('.gitignore', 'r') as f:
        lines = f.readlines()
    
    with open('.gitignore', 'w') as f:
        for line in lines:
            if 'results/' not in line:
                f.write(line)

unignore_results()
