import argparse
import numpy

import cPickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("dataset",
                    type=argparse.FileType('r'),
                    help="location of the textfile")
parser.add_argument("dictionary",
                    default="dictionary.pkl",
                    help="location of dictionary you created")
parser.add_argument("output_name",
                    default="train.pkl",
                    help="location of output")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.dictionary) as f:
        dictionary = pkl.load(f)
    text = []
    for lines in args.dataset:
        sentence = lines.lower().strip().split(' ')
        text.append([dictionary.get(word, 1) for word in sentence])
    with open(args.output_name, 'w') as f:
        pkl.dump(text, f)
