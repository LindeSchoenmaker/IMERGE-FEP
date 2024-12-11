import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from alchemlyb import concat as alchemlyb_concat
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.gmx import extract_u_nk
from alchemlyb.visualisation import plot_convergence, plot_mbar_overlap_matrix

parser = argparse.ArgumentParser()
parser.add_argument("-p",
                    "--path",
                    help="directory containing dhdl.xvg files",
                    type=str,
                    default="workpath")
parser.add_argument("-c",
                    "--condition",
                    help="RHFE or RBFE, which legs to compare",
                    type=str,
                    choices=['rhfe', 'rbfe'])
parser.add_argument("-pl",
                    "--plot",
                    help="plot convergence figure",
                    type=bool,
                    default = False)
parser.add_argument("-r",
                    "--runs",
                    help="which runs to use",
                    default=['run1', 'run2', 'run3'],
                    nargs='+')
parser.add_argument("-e",
                    "--edges",
                    help="which edges to use",
                    nargs='+')

def get_dg(df):
    """Read dg from result table and convert from kJ to kCal"""
    dg_kJ = df.iloc[0, len(df)-1]
    df_kCal = dg_kJ / 4.2
    return df_kCal



if __name__ == "__main__":
    args = parser.parse_args()

    dg_dict = defaultdict(lambda: defaultdict(dict))
    ddg_dict = {}
    ddg_err_dict = {}
    ddg_mbar_err_dict = {}

    dg_err_dict = defaultdict(lambda: defaultdict(dict))

    base_path = args.path
    if args.edges:
        edges = args.edges
    else:
        edges = [os.path.basename(x) for x in glob.glob(f'{base_path}/*') if 'edge' in x]

    condition = args.condition # ['vacuum', 'water']
    if condition == 'rhfe':
        legs = ['water', 'vacuum']
    elif condition == 'rbfe':
        legs = ['protein', 'water']


    plot = args.plot
    runs = args.runs

    for edge in edges:
        for leg in legs:
            for run in runs:
                print(edge, run, leg)
                numbers = [os.path.basename(x) for x in glob.glob(f'{base_path}/{edge}/{leg}/*')]
                data_AB = [glob.glob(f'{base_path}/{edge}/{leg}/{number}/{run}/production/dhdl.xvg')[0] for number in numbers if int(number) < 15]
                data_list_AB = [extract_u_nk(xvg, T=298.15) for xvg in data_AB]
                mbar_AB = MBAR().fit(alchemlyb_concat(data_list_AB))
                print('B')
                data_B = [glob.glob(f'{base_path}/{edge}/{leg}/{number}/{run}/production/dhdl.xvg')[0] for number in numbers if int(number) >= 15]
                data_list_B = [extract_u_nk(xvg, T=298.15) for xvg in data_B]
                mbar_B = MBAR().fit(alchemlyb_concat(data_list_B))

                dg_AB = get_dg(mbar_AB.delta_f_)
                dg_B = get_dg(mbar_B.delta_f_)
                dg = dg_AB + dg_B
                dg_dict[edge][leg][run] = (dg_AB, dg_B, dg)

                err = get_dg(mbar_AB.d_delta_f_) + get_dg(mbar_B.d_delta_f_)
                dg_err_dict[edge][leg][run] = err

            if plot:
                df = forward_backward_convergence(data_list_AB, 'mbar')
                fig, axs = plt.subplots(2, 2, figsize=(16,9))
                plot_convergence(df, ax=axs[0,0], units = 'kJ/mol')
                plot_mbar_overlap_matrix(mbar_AB.overlap_matrix, ax=axs[0,1])

                df = forward_backward_convergence(data_list_B, 'mbar')
                plot_convergence(df, ax=axs[1,0], units = 'kJ/mol')
                plot_mbar_overlap_matrix(mbar_B.overlap_matrix, ax=axs[1,1])

                if not os.path.exists('figure'):
                    os.makedirs('figure')
                fig.savefig(f'figure/{edge}_{leg}.png')
            dg_dict[edge][leg]['average'] =  sum([dg_dict[edge][leg][run][-1] for run in runs])/len(runs)
            dg_dict[edge][leg]['sd'] =  np.std([dg_dict[edge][leg][run][-1] for run in runs])
            dg_err_dict[edge][leg]['average'] =  sum([dg_err_dict[edge][leg][run] for run in runs])/len(runs)

        ddg_dict[edge] = dg_dict[edge][legs[0]]['average'] - dg_dict[edge][legs[1]]['average']
        ddg_err_dict[edge] = math.sqrt(dg_dict[edge][legs[0]]['sd']**2 + dg_dict[edge][legs[1]]['sd']**2)
        ddg_mbar_err_dict[edge] = dg_err_dict[edge][legs[0]]['average'] + dg_err_dict[edge][legs[1]]['average']


    print(ddg_dict)
    print(ddg_err_dict)
    print(ddg_mbar_err_dict)

    with open("ddg_err.json", "w") as outfile: 
        json.dump(ddg_err_dict, outfile)

    with open("ddg.json", "w") as outfile: 
        json.dump(ddg_dict, outfile)

    with open("dg.json", "w") as outfile: 
        json.dump(dg_dict, outfile)

    with open("dg_err.json", "w") as outfile: 
        json.dump(dg_err_dict, outfile)