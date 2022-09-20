import argparse
from imageNet_protocols import protocols as P
from pathlib import Path


def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='The script generates the csv-files that are necessary for generating the datasets associated with the protocols.'
    )

    data_root_default = "/local/scratch/datasets/ImageNet/ILSVRC2012"
    csv_root_default = Path(
        __file__).parent.parent.parent.parent.parent.resolve() / 'data/csv'

    parser.add_argument('-d', '--data_path', type=str, nargs='?', default=data_root_default,
                        const=data_root_default, help='filesystem path to root directory of the ILSVRC2012 dataset')
    parser.add_argument('-o', '--out_path', type=str, nargs='?', default=csv_root_default, const=csv_root_default,
                        help='filesystem path to directory where generated csv-files should be stored in')
    parser.add_argument('-p', '--protocols', type=int, nargs='+', default=[1, 2, 3],
                        help='protocols for which csv files are generated, must be a subset of {1,2,3}')
    return parser


def generate_csv(data_path, out_path, protocols):
    P.generate_csv(data_path=data_path,
                   out_path=out_path, protocols=protocols)


if __name__ == "__main__":
    parser = command_line_options()
    args = parser.parse_args()

    generate_csv(args.data_path, args.out_path, args.protocols)
