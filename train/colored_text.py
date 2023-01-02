from simple_colors import *
import csv 

def get_token_representation(text, start_sal_tokens=['<sal0>', '<sal1>'] , end_sal_tokens=['</sal0>', '</sal1>']):
    tokens = [''] + text.split()
    current_class = -1
    token_class = []
    for idx, token in enumerate(tokens):
        if token in start_sal_tokens:
            current_class = start_sal_tokens.index(token)
        elif token in end_sal_tokens:
            current_class = -1
        else:
            token_class.append(current_class)
    word_tokens = [t for t in text.split() if t not in start_sal_tokens + end_sal_tokens]
    return token_class, word_tokens

def print_colored(token_class, tokens):
    html = ""
    for idx, token in enumerate(tokens):
        current_class = token_class[idx]
        # red - 0
        # blue - 1
        if current_class == 0:
            color = "#d90429"
        elif current_class == 1:
            color = "#0077b6"
        else:
            color = "#FFFFFF"
        html += "<span style='background-color:" + color + "'>&nbsp;" + token + "&nbsp;</span>"

    return html

def print_difference(token_class, original_token_class, tokens):
    html = ""
    for idx, token in enumerate(tokens):
        if token_class[idx] != original_token_class[idx]:
            color = "#39e75f"
        else:
            color = "#FFFFFF"
        html += "<span style='background-color:" + color + "'>&nbsp;" + token + "&nbsp;</span>"
    return html


if __name__ == "__main__":
    with open('local_search.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        out_file = open('out.html', 'w')
        out_file.write('<html>')
        annotation_types = ["Expert-Adaptive", "Adaptive", "Double", "Human", "Single"]
        for row in csv_reader:
            if line_count != 0:
                # generate new and original regresentations
                original_representation, word_tokens = get_token_representation(row[0])
                new_representation, _ = get_token_representation(row[2]) 

                # 4 conditions for each text sample
                if line_count % 5 == 1:
                    out_file.write('<br><br>')
                    example_number = int(line_count/5 + 1)
                    out_file.write('<b>%s</b>' % "Example #"+str(example_number))

                example_type = (line_count + 4) % 5
                out_file.write('<br>')
                out_file.write('<b>%s</b>' % f"{annotation_types[example_type]} Text:")
                out_file.write(print_colored(original_representation, word_tokens))
                out_file.write('<br>')
                out_file.write('<b>%s</b>' % "Modified Text:")
                out_file.write(print_colored(new_representation, word_tokens))
                out_file.write('<br>')
                out_file.write('<b>%s</b>' % "Difference Text:")
                out_file.write(print_difference(new_representation, original_representation, word_tokens))
                out_file.write('<br><br>')

                out_file.write('<b>%s</b>' % "Original Text Correctness Prediction: " + str((1 - (float)(row[1]))))
                out_file.write('<br>')
                out_file.write('<b>%s</b>' % "Modified Text Correctness Prediction: " + str((1 - (float)(row[3]))))
                out_file.write('<br><br><br><br>')
            line_count += 1
        out_file.write('</html>')
        out_file.close()
