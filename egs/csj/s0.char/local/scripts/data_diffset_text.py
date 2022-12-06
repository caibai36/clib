import argparse
import os

parser = argparse.ArgumentParser(description="Get the result dataset by removing sentences of text scp file from the original dataset.\ne.g. python local/scripts/data_diffset_text.py --original data/test --remove data/train/text data/dev/text --result tmp", formatter_class= argparse.RawTextHelpFormatter)
parser.add_argument("--original", type=str, required=True, default=None,
                    help="the original dataset dir (eg. ./data/test), which might contains scp files such as text, utt2gender, utt2spk, wav.scp, segments sharing same uttids.")
parser.add_argument("--remove", type=str, nargs="+", required=True, default=None, help="the text scp files to move from original set")
parser.add_argument("--result", type=str, required=True, default=None, help="the output directory of updated dataset filtering out the sentences at the remove text files")
args = parser.parse_args()

remove_text_set = set()
for text_file in args.remove:
    with open(text_file, 'r', encoding='utf8') as f_remove:
        for line in f_remove:
            uttid, content = line.strip().split(" ", maxsplit=1)
            remove_text_set.add(content)

preserved_uttids = set()
original_text_file = os.path.join(args.original, "text")
with open(original_text_file, 'r', encoding='utf8') as f_original:
    for line in f_original:
        uttid, content = line.strip().split(" ", maxsplit=1)
        if content not in remove_text_set:
            preserved_uttids.add(uttid)

if (not os.path.exists(args.result)): os.makedirs(args.result)
for file in ["text", "utt2gender", "utt2spk", "wav.scp", "segments"]:
    input_file = os.path.join(args.original, file)
    output_file = os.path.join(args.result, file)

    if (os.path.exists(input_file)):
        with open(input_file, 'r', encoding='utf8') as f_input, \
             open(output_file, 'w', encoding='utf8') as f_output:
                  for line in f_input:
                      uttid, content = line.strip().split(" ", maxsplit=1)
                      if uttid in preserved_uttids:
                          f_output.write(f"{uttid} {content}\n")
    else:
        print(f"Warning: {file} not exist in {args.original}")
