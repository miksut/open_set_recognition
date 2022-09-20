from context import approaches, architectures, data_prep, tools
from datetime import datetime
import imageNet_protocols
import logging
from pathlib import Path
import pickle
from settings import *
import sys
import torch
from torch.utils.data import DataLoader

# instantiate module logger
logger = logging.getLogger('eosa.experiments.eval_algos')

args = tools.command_line_options(
    MODELS, MODEL_OUT_PATH_DEFAULT, OSCR_PATH_DEFAULT, LOG_PATH_DEFAULT, DATA_PATH_DEFAULT, CSV_PATH_DEFAULT, KU_TARGET_DEFAULT, UU_TARGET_DEFAULT, NUM_WORKERS, MODEL_IN_BUILD_DNN_TEST, MODEL_IN_USE_DNN_TEST, DNN_FEATURES, EVM_HYPERPARAMS, OPENMAX_HYPERPARAMS, APPROACHES, FPR_THRESHOLDS, MODEL_EXTENDABLE, PROSER_HYPERPARAMS, MODEL_IN_EXTEND_DNN_TEST, OPTIMIZER_PARAMS)

# configure and add file handler to project top level logger
file_name_out = datetime.now().strftime(
    "%d-%m-%Y_%H:%M:%S") + "_" + (sys.argv[0].split("/")[-1]) + "_" + "_".join(sys.argv[1:]) + '.log'
file_handler = logging.FileHandler(LOG_PATH_DEFAULT / file_name_out, mode='w')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(message)s', datefmt='%d-%m-%Y_%H:%M:%S')
file_handler.setFormatter(file_formatter)
logging.getLogger('eosa').addHandler(file_handler)

logger.debug('\n')
logger.info('Argument parser attributes:')
logger.debug('############################################################')
logger.info(args)


def generate_datasets(args, return_paths_only=False):
    # generate dict of all available csv-files and their path for the current protocol
    csv_files = {str(Path(v.name).stem).replace(f'_p{args.protocol}', '').replace(
        f'p{args.protocol}', ''): v for v in sorted(args.csv_path.glob(f'*{args.protocol}*'))}

    if args.procedure == 'train':
        train_cls = sorted(set(['train_' + i for i in args.train_cls]))
        val_cls = sorted(set(['val_' + i for i in args.val_cls]))

        if return_paths_only:
            train_data_path = csv_files['train'] if len(
                train_cls) > 1 else csv_files[train_cls[-1]]
            val_data_path = csv_files['val'] if len(
                val_cls) > 1 else csv_files[val_cls[-1]]
            return train_data_path, val_data_path

        train_set, val_set, _, msgs = imageNet_protocols.tools.generate_datasets(args.data_path, csv_files['train'] if len(
            train_cls) > 1 else csv_files[train_cls[-1]], csv_files['val'] if len(val_cls) > 1 else csv_files[val_cls[-1]], None)
        train_loader = DataLoader(
            train_set, shuffle=True, batch_size=args.batch, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, shuffle=False,
                                batch_size=args.batch, num_workers=args.num_workers)
        logger.info(msgs[0])
        logger.info(msgs[1])
        return train_set, val_set, train_loader, val_loader

    elif args.procedure == 'test':
        # TODO: the test dataset is hardcoded to consist of KKCs and UUCs. Add customisation options
        if return_paths_only:
            return csv_files['test_kk_uu']

        _, _, test_set, msgs = imageNet_protocols.tools.generate_datasets(
            args.data_path, None, None, csv_files['test_kk_uu'])
        test_loader = DataLoader(
            test_set, shuffle=False, batch_size=args.batch, num_workers=args.num_workers)
        logger.info(msgs[2])
        return test_set, test_loader


if args.approach in ['base', 'entropic']:

    if args.procedure == 'train':
        train_set, val_set, train_loader, val_loader = generate_datasets(args)
        approach = getattr(approaches, APPROACHES[args.approach])(args.protocol, args.gpu, args.ku_target, args.uu_target, args.model_out_path,
                                                                  args.log_path, args.oscr_path, args.train_cls, args.architecture, args.feature_dimension, train_set.num_kk_classes, args.optimizer, args.epochs)
        approach.train(train_loader, val_loader)

    elif args.procedure == 'test':

        model_dict = torch.load(args.model_to_test)
        test_set, test_loader = generate_datasets(args)

        approach = getattr(approaches, model_dict['approach_train'])(
            *model_dict['instance'].values())
        approach.test(test_loader=test_loader,
                      model_dict=model_dict)

elif args.approach in ['evm', 'openmax']:

    dnn_dict = torch.load(args.dnn_features)

    logger.debug('\n')
    logger.info(
        f'Info on loaded feature-extracting model: {dnn_dict["model_name"]}, best trainings performance {dnn_dict["eval_metric_opt"]:.6f} ({dnn_dict["eval_metric"]}) achieved in epoch {dnn_dict["epoch_opt"]}')

    pretrained_dnn = getattr(architectures, dnn_dict['instance']['architecture'])(
        feature_dim=dnn_dict['instance']['df_dim'], num_classes=dnn_dict['instance']['num_cls'])
    pretrained_dnn.load_state_dict(
        dnn_dict['state_dict'])

    extractor = getattr(data_prep, f'{APPROACHES[args.approach]}Extractor')(
        pretrained_dnn, args.gpu, APPROACHES[args.approach])

    if args.procedure == 'train':
        train_set, val_set, train_loader, val_loader = generate_datasets(args)

        # option for loading the data dict to avoid time consuming extraction and conversion process
        if args.load_data_dict:
            train_path, val_path = generate_datasets(
                args, return_paths_only=True)
            path_stem = Path(train_path).parent.parent / Path('dicts')

            train_feat = extractor.load_data_dict(
                str(path_stem / Path(train_path).stem) + f'_features_dict_{Path(args.dnn_features).stem}.pkl')['data']
            pos_classes = extractor.load_data_dict(
                str(path_stem / Path(train_path).stem) + f'_features_dict_{Path(args.dnn_features).stem}.pkl')['pos_classes']
            val_feat = extractor.load_data_dict(
                str(path_stem / Path(val_path).stem) + f'_features_dict_{Path(args.dnn_features).stem}.pkl')['data']
            val_logits = extractor.load_data_dict(
                str(path_stem / Path(val_path).stem) + f'_logits_dict_{Path(args.dnn_features).stem}.pkl')['data']

        else:
            train_feat, pos_classes = extractor.extract_train_features(
                train_loader)
            val_feat, val_logits = extractor.extract_features_and_logits(
                val_loader)
            train_path, val_path = generate_datasets(
                args, return_paths_only=True)
            path_stem = Path(train_path).parent.parent / Path('dicts/')

            extractor.store_data_dict(train_feat, str(path_stem / Path(train_path).stem) +
                                      f'_features_dict_{Path(args.dnn_features).stem}', pos_classes=pos_classes)
            extractor.store_data_dict(val_feat, str(path_stem / Path(val_path).stem) +
                                      f'_features_dict_{Path(args.dnn_features).stem}')
            extractor.store_data_dict(val_logits, str(path_stem / Path(val_path).stem) +
                                      f'_logits_dict_{Path(args.dnn_features).stem}')

        approach = getattr(approaches, APPROACHES[args.approach])(args.protocol, args.gpu, args.ku_target,
                                                                  args.uu_target, args.model_out_path, args.log_path, args.oscr_path, args.train_cls, args.architecture, args.dnn_features.stem, args.fpr_thresholds)
        approach.train(pos_classes, train_feat,
                       val_feat, val_logits, args.hyperparameters)

    elif args.procedure == 'test':

        file_handler = open(args.model_to_test, 'rb')
        model_dict = pickle.load(file_handler)

        test_set, test_loader = generate_datasets(args)

        # option for loading the data dict to avoid time consuming extraction and conversion process
        if args.load_data_dict:
            test_path = generate_datasets(
                args, return_paths_only=True)
            path_stem = Path(test_path).parent.parent / Path('dicts/')

            test_feat = extractor.load_data_dict(
                str(path_stem / Path(test_path).stem) + f'_features_dict_{Path(args.dnn_features).stem}.pkl')['data']
            test_logits = extractor.load_data_dict(
                str(path_stem / Path(test_path).stem) + f'_logits_dict_{Path(args.dnn_features).stem}.pkl')['data']

        else:
            test_feat, test_logits = extractor.extract_features_and_logits(
                test_loader)
            test_path = generate_datasets(
                args, return_paths_only=True)
            path_stem = Path(test_path).parent.parent / Path('dicts/')

            extractor.store_data_dict(test_feat, str(path_stem / Path(test_path).stem) +
                                      f'_features_dict_{Path(args.dnn_features).stem}')
            extractor.store_data_dict(test_logits, str(path_stem / Path(test_path).stem) +
                                      f'_logits_dict_{Path(args.dnn_features).stem}')

        approach = getattr(approaches, model_dict['approach_train'])(
            *model_dict['instance'].values())

        approach.test(test_feat, test_logits, args.hyperparameters, model_dict)

elif args.approach in ['proser']:

    if args.procedure == 'train':
        train_set, val_set, train_loader, val_loader = generate_datasets(args)

        for idx in range(len(args.no_dummy_clfs)):

            basis_model_dict = torch.load(args.model_extendable)

            approach = getattr(approaches, APPROACHES[args.approach])(args.protocol, args.gpu, args.ku_target, args.uu_target, args.model_out_path,
                                                                      args.log_path, args.oscr_path, args.train_cls, args.architecture, args.epochs, basis_model_dict, args.lambda1, args.lambda2, args.alpha, args.no_dummy_clfs[idx], args.bias_computation)
            approach.train(train_loader, val_loader)

    elif args.procedure == 'test':
        model_dict = torch.load(args.model_to_test)
        test_set, test_loader = generate_datasets(args)

        approach = getattr(approaches, model_dict['approach_train'])(
            *model_dict['instance'].values())
        approach.test(test_loader=test_loader,
                      model_dict=model_dict)
