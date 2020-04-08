#!/usr/bin/env python3

import json
import sys


def main():
    assert len(sys.argv) == 3

    with open(sys.argv[1]) as f1:
        with open(sys.argv[2]) as f2:
            print(f"{sys.argv[1]} equals {sys.argv[2]}: \n{json.load(f1) == json.load(f2)}")


if __name__ == "__main__":
    main()
