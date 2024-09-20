import sys

if __name__ == '__main__':
    source = sys.argv[1]
    destination = sys.argv[2]
    b_num = int(sys.argv[3]) # 6 for PTB, 7 for multinli
    e_num = int(sys.argv[4]) # 1 for PTB, 1 for multinli

    with open(source, 'r') as source_file:
        with open(destination, 'w') as destination_file:
            for line in source_file:
                destination_file.write(line[b_num:-(e_num+1)]+"\n")