from context import tools
import matplotlib.pyplot as plt
from pathlib import Path


acronym_to_name = {'baseline': "Softmax",
                   'entropic': 'Entropic Open Set Loss',
                   'openmax': 'OpenMax',
                   'evm': 'EVM',
                   'proser': 'PROSER'}

colors_base = {1: '#8FBBD9', 2: '#1F77B4', 3: '#0F3B5A'}
colors_eos = {1: '#FFBF86', 2: '#FF7F0E', 3: '#7F3F07'}
colors_openmax = {1: '#95CF95', 2: '#2CA02C', 3: '#165016'}
colors_evm = {1: '#EA9393', 2: '#D62728', 3: '#6B1314'}
colors_proser = {1: '#C9B3DE', 2: '#9467BD', 3: '#4A335E'}

colors = {'Softmax': colors_base, 'Entropic Open Set Loss': colors_eos,
          'OpenMax': colors_openmax, 'EVM': colors_evm, 'PROSER': colors_proser}

oscr_path_root = Path(
    __file__).parent.parent.parent.parent.parent.resolve() / 'results' / 'oscr'


"""
############################################################
 CONFIGURATIONS
############################################################

In the list 'oscr_path_model', the filesystem paths to the csv-files holding the OSCR-values for
the plotted models must be defined. Note that in order to work as intended, the script assumes a
directory structure as follows:
 - master_suter/results/oscr/protocol_<PROT>/<APPROACH>/<CSV_FILENAME>.csv

Further, assign the desired name of the output png file to the variable 'figure_name'.

"""

oscr_path_model = [
    # Baseline
    # --------------------------------------------------
    'protocol_1/baseline/p1_baseline_test_p1_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv',
    'protocol_2/baseline/p2_baseline_test_p2_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv',
    'protocol_3/baseline/p3_baseline_test_p3_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv'

    # Entropic Open Set
    # --------------------------------------------------
    # 'protocol_1/entropic/p1_entropicosl_test_p1_traincls(kk+ku)_entropicosl_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv',
    # 'protocol_2/entropic/p2_entropicosl_test_p2_traincls(kk+ku)_entropicosl_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv',
    # 'protocol_3/entropic/p3_entropicosl_test_p3_traincls(kk+ku)_entropicosl_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv'

    # OpenMax
    # --------------------------------------------------
    # 'protocol_1/openmax/test/p1_openmax_test_TS_1000.0_DM_2.00_alpha5_cosine_oscr_values_dnn_p1_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999).csv',
    # 'protocol_2/openmax/test/p2_openmax_test_TS_1000.0_DM_2.00_alpha5_cosine_oscr_values_dnn_OPT_p2_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999).csv',
    # 'protocol_3/openmax/test/p3_openmax_test_TS_1000.0_DM_2.00_alpha5_cosine_oscr_values_dnn_p3_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999).csv'

    # EVM
    # --------------------------------------------------
    # 'protocol_1/evm/test/p1_evm_test_TS_1000_DM_0.50_CT_1.00_cosine_oscr_values_dnn_p1_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999).csv',
    # 'protocol_2/evm/test/p2_evm_test_TS_75_DM_0.20_CT_1.00_cosine_oscr_values_dnn_OPT_p2_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999).csv',
    # 'protocol_3/evm/test/p3_evm_test_TS_150_DM_0.40_CT_1.00_cosine_oscr_values_dnn_p3_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999).csv'

    # PROSER
    # --------------------------------------------------
    # 'protocol_1/proser/p1_proser_test_p1_traincls(kk)_proser_dummy5_epochsfine10_λ(1.0+1.0)_α1.0_bias(True)_basis_p1_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv',
    # 'protocol_2/proser/p2_proser_test_p2_traincls(kk)_proser_dummy3_epochsfine10_λ(1.0+1.0)_α1.0_bias(True)_basis_p2_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv',
    # 'protocol_3/proser/p3_proser_test_p3_traincls(kk)_proser_dummy20_epochsfine10_λ(1.0+1.0)_α1.0_bias(True)_basis_p3_traincls(kk)_baseline_resnet50_feature_df512_e200_optAdam(0.001+0.7+0.999)_oscr_values.csv'
]

figure_name = 'softmax_comparison.pdf'


plt.figure()
for idx, model in enumerate(oscr_path_model):
    data = tools.load_oscr_metrics(oscr_path_root / model)
    label = acronym_to_name[model.split('/')[1]]
    plt.semilogx(data['fpr'], data['ccr'],
                 label=label+f' (Protocol {idx+1})', color=colors[label][idx+1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('Correct Classification Rate (CCR)')
plt.tight_layout()
plt.legend()
plt.savefig(oscr_path_root / figure_name)
