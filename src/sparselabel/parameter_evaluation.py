import argparse
import json
import math

import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

from sparselabel.constants import ENCODING


# pylint: disable=too-many-locals
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', help="file_paths for the files that should be summarized")
    parser.add_argument('-o', help="Path to save the output csv to")
    parser.add_argument('-p', help="parameter to analyze")
    parser.add_argument('-l', help="label for parameter")
    parser.add_argument('-u', help="unit for parameter")
    args = parser.parse_args()

    with open(args.file, encoding=ENCODING) as fp:
        experiments = json.load(fp)

    parameter = [experiment[args.p] for experiment in experiments if (experiment[args.p] is not None) and experiment[args.p] != math.inf]
    parameter_manual = [experiment[f"{args.p}_manual"] for experiment in experiments if (experiment[args.p] is not None) and experiment[args.p] != math.inf]
    identifiers = [experiment["identifier"] for experiment in experiments if (experiment[args.p] is not None) and experiment[args.p] != math.inf]

    num_slices = len(parameter)
    exam = list(range(num_slices)) * 2
    judge = ["A"] * num_slices + ["B"] * num_slices
    rating = list(parameter) + list(parameter_manual)

    df = pd.DataFrame({'exam': exam,
                       'judge': judge,
                       'rating': rating})
    icc = pg.intraclass_corr(data=df, targets='exam', raters='judge', ratings='rating')

    icc.set_index('Type')
    print(args.file)
    print(args.p)
    print(icc)

    distances = np.array([parameter_manual, parameter])
    mean_value_of_two_ratings = np.mean(distances, axis=0)
    difference_between_two_ratings = distances[0] - distances[1]

    max_diffs = np.argsort(np.abs(difference_between_two_ratings))[-2:][::-1]
    print(f"cases with max difference = {identifiers[max_diffs[0]]}, {identifiers[max_diffs[1]]}")

    plt.scatter(mean_value_of_two_ratings, difference_between_two_ratings)

    std = np.std(difference_between_two_ratings)
    mean = np.mean(difference_between_two_ratings)

    print(f"mean: {mean}")

    plt.axhline(mean, color='gray', linestyle='--', lw=0.4)
    plt.axhline(mean + 1.96 * std, color='gray', linestyle='--', lw=0.4)
    plt.axhline(mean - 1.96 * std, color='gray', linestyle='--', lw=0.4)
    plt.xlabel(f"Mean {args.l} {args.u}")
    plt.ylabel(f'Expert {args.l} - Model {args.l} {args.u}')
    # plt.show()
    plt.savefig(f"{args.o}/{args.p}.png")


if __name__ == "__main__":
    main()
