import argparse

def main():
    parser = argparse.ArgumentParser(description="Find the sentences containing punctuation marks to analyze",
                                     epilog = 'usage: python find_mark.py  --in_text="text" --in_mark="mark"')
    parser.add_argument("--in_text", type=str, default="stext", help="input text file including all sentences")
    parser.add_argument("--in_mark", type=str, default="mark", help="mark file including punctuation marks to find")
    parser.add_argument("--num_lines", type=int, default="10", help="the maximum lines to be print")
    args = parser.parse_args()
    
    in_text = args.in_text
    in_mark = args.in_mark
    num_lines = args.num_lines

    texts = []
    with open(in_text) as f_text:
        for line in f_text:
            texts.append(line.strip())

    with open(in_mark) as f_mark:
        for mark in f_mark:
            mark = mark.strip()
            print(mark)
            counter = 0
            for line in texts:
                if mark in line:
                    print(line)
                    counter += 1
                    if counter == num_lines:
                        break
            print()

if __name__ == "__main__":
    main()
