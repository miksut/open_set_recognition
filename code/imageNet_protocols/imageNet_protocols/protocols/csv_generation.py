import pandas as pd
from pathlib import Path
from . import protocol1, protocol2, protocol3


def generate_csv(data_path, out_path, protocols=[1, 2, 3]):
    """
    :param data_path: String, filesystem path to root directory of the ILSVRC2012 dataset
    :param out_path: String, filesystem path to  directory where generated csv-files should be stored in
    :param protocols: List of Integers, protocols for which csv files are generated (default: all)
    """

    assert isinstance(
        protocols, list), 'Argument \'protocols\' must be a list of integers'
    for i in protocols:
        assert i in [1, 2, 3], 'Available protocols: 1, 2, 3'

    prot = {1: protocol1, 2: protocol2, 3: protocol3}

    path = Path(out_path)
    path.mkdir(parents=True, exist_ok=True)

    # path to ImageNet files used by protocols
    info_path = Path(protocol1.__file__).parent.parent / 'access_files'

    for i in protocols:
        print(f'\nPreparing CSV files for Protocol {i}')
        print("##################################################\n")
        getattr(prot[i], f'generate_csv_p{i}')(
            data_path=data_path, info_path=info_path, out_path=out_path)

        df_kkc = pd.read_csv(out_path / f'test_kkp{i}.csv', header=None)
        df_uuc = pd.read_csv(out_path / f'test_uup{i}.csv', header=None)
        df_cat = pd.concat([df_kkc, df_uuc], axis=0)

        file_path_out = out_path / \
            f'test_kkp{i}_uup{i}.csv'

        df_cat.to_csv(file_path_out, index=False, header=False)
