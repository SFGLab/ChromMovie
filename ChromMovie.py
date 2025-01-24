#!/usr/bin/env python3
#################################################################
########### Krzysztof Banecki, Warsaw 2025 ######################
#################################################################

from run_ChromMovie_from_yaml import ChromMovie_from_yaml
import argparse


def main():
    parser = argparse.ArgumentParser(description="A script to process user inputs for ChromMovie.")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Folder containing input scHi-C contacts in csv format."
    )

    args = parser.parse_args()
    ChromMovie_from_yaml(args.input)


if __name__ == "__main__":
    main()
