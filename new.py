import argparse

parser = argparse.ArgumentParser(description='Sample Argument Parser')
parser.add_argument('-m', '--mode', help='Mode of operation')
parser.add_argument('-p', '--path', help='File path')
args = parser.parse_args()

mode = args.mode
path = args.path

print('Mode:', mode)
print('Path:', path)
print('Done!')