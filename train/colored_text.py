from simple_colors import *
import csv 

def print_colored(text, sal_token='<in_sal>', no_sal_token='<out_sal>', cp_token='<cp>'):
    tokens = [''] + text.split()
    for idx, token in enumerate(tokens):
        if idx > 0:
            if token in [sal_token, no_sal_token, cp_token]:
                continue
            if tokens[idx - 1] == sal_token:
                print(red(token), end=" ")
            elif tokens[idx - 1] == cp_token:
                print(blue(token), end=" ")
            else:
                print(token, end=" ")
    print()


if __name__ == "__main__":
    a = 0
    with open('local_search.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                a += 1
                print("Original Text:", end=" ")
                print_colored(row[0])
                print("Modified Text:", end=" ")
                print_colored(row[2])
                print("Original Text Correctness Prediction:", 1 - (float)(row[1]))
                print("Modified Text Correctness Prediction:", 1 - (float)(row[3]))
                print()
                print()
            line_count += 1
    print(a)
