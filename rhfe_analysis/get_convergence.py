import argparse
import glob
import json
import os
from collections import defaultdict

import alchemlyb
import numpy as np
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.gmx import extract_u_nk

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

def get_estimate_uncertainty(data_list, slice_fw = -1, slice_bw = 0):
    u_nk = alchemlyb.concat([data[-slice_bw:slice_fw] for data in data_list])
    estimate = MBAR().fit(u_nk)
    return estimate.delta_f_.iloc[0,-1] / 4.2, estimate.d_delta_f_.iloc[0,-1] / 4.2


if __name__ == "__main__":
    args = parser.parse_args()
    base_path = args.path

    if args.edges:
        edges = args.edges
    else:
        edges = [os.path.basename(x) for x in glob.glob(f'{base_path}/*') if 'edge' in x]
    print(edges)

    condition = args.condition # ['vacuum', 'water']

    if condition == 'rhfe':
        legs = ['water', 'vacuum']

    elif condition == 'rbfe':
        legs = ['protein', 'water']

    runs = args.runs

    conv_dict = defaultdict(dict)
    conv_std_dict = defaultdict(dict)

    interval = 10
    max_change = 0.1 #kcal/mol

    for edge in edges:
        print(edge)
        forward = defaultdict(list)
        forward_error = defaultdict(list)
        forward_convergence = []
        for leg in legs:
            forward_convergence = []
            numbers = [os.path.basename(x) for x in glob.glob(f'{base_path}/{edge}/{leg}/*')]
            for run in runs:
                last_value = None
                data_AB = [glob.glob(f'{base_path}/{edge}/{leg}/{number}/{run}/production/dhdl.xvg')[0] for number in numbers if int(number) < 15]
                data_list_AB = [extract_u_nk(xvg, T=298.15) for xvg in data_AB]

                data_B = [glob.glob(f'{base_path}/{edge}/{leg}/{number}/{run}/production/dhdl.xvg')[0] for number in numbers if int(number) >= 15]
                data_list_B = [extract_u_nk(xvg, T=298.15) for xvg in data_B]

                # mbar_AB = MBAR().fit(alchemlyb_concat(data_list_AB))
                # mbar_B = MBAR().fit(alchemlyb_concat(data_list_B))

                # dg_AB = get_dg(mbar_AB.delta_f_)
                # dg_B = get_dg(mbar_B.delta_f_)

                # final_dg = dg_AB + dg_B
                # err = get_dg(mbar_AB.d_delta_f_) + get_dg(mbar_B.d_delta_f_)
                count = 0
                for i in range(1, int(len(data_list_AB[0])/interval) + 1):
                    # Do the forward
                    slice = interval * i
                    forward_AB, forward_error_AB = get_estimate_uncertainty(data_list_AB, slice_fw = slice)
                    forward_B, forward_error_B = get_estimate_uncertainty(data_list_B, slice_fw = slice)
                    forward[leg + run].append(forward_AB + forward_B)
                    forward_error[leg + run].append(forward_error_AB + forward_error_B)
                    curr_value = forward_AB + forward_B
                    if last_value:
                        if abs(last_value-curr_value) < max_change:
                            count += 1
                        else:
                            count = 0
                            last_value = curr_value
                        if count >= 20: #converged if stays within window for 10 * interval
                            forward_convergence.append(i - 20)
                            break
                        if i == int(len(data_list_AB[0])/interval):
                            forward_convergence.append(i - count)
                            break
                    else:
                        last_value = curr_value
            conv_dict[edge][leg] =  sum(forward_convergence)/len(forward_convergence)
            conv_std_dict[edge][leg] =  np.std(forward_convergence)
        
    with open(f"conv_{condition}.json", "w") as outfile: 
        json.dump(conv_dict, outfile)

    with open(f"conv_std_{condition}.json", "w") as outfile: 
        json.dump(conv_std_dict, outfile)