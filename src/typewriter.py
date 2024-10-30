from time import sleep

def typewriter(file_path, cps = 1):
    with open(file_path, 'r') as f:
        content = f.read()

    for char in content:
        print(char, end='', flush=True)
        sleep(1 / cps)

if __name__ == '__main__':
    import sys

    file = sys.argv[1]
    cps = int(sys.argv[2])

    typewriter(file, cps)

