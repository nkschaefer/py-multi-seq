#! /usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

def main(args):
    if len(args) < 2:
        print("USAGE: assign_barcodes.py <counts file> (max_cells_per_droplet)", file=sys.stderr)
        print("max_cells_per_droplet (OPTIONAL, default = 2) is the maximum number of cells \
that can be inferred within a single droplet before it is determined to be \"Negative.\" \
For example if 2 (default), cells inferred to be a mixture of 2 barcodes will be marked as \
doublets of the likeliest source barcodes, and cells inferred to be a mixture of 3 or more \
barcodes will be considered noise / negative droplets.", file=sys.stderr)
        exit(1)
    
    max_cells = 2
    if len(args) > 2:
        max_cells = int(args[2])

    counts = pd.read_csv(args[1], sep="\t", index_col='barcode')
   
    # Drop columns where everything is zero
    counts = counts.loc[counts.sum(axis=1) > 0, :]

    assignments = []

    for idx in range(0, counts.shape[0]):
        vals = counts.iloc[idx,:]
        vals.sort_values(ascending=False, inplace=True)
        n_nzero = np.sum(vals > 0)
        ll_max = None
        nchosen_max = None
        ll_second = None
        ll_max_nonneg = None

        for nchosen in range(1, n_nzero+1):
            mean_chosen = np.mean(vals[0:nchosen])
            mean_err = 0.0
            if nchosen < n_nzero:
                mean_err = np.mean(vals[nchosen:n_nzero+1])
            
            ll = 0.0
            for val in vals[0:nchosen]:
                ll += poisson.logpmf(val, mean_chosen)
            for val in vals[nchosen:n_nzero+1]:
                ll += poisson.logpmf(val, mean_err)
            
            if nchosen <= max_cells:
                if ll_max_nonneg is None or ll > ll_max_nonneg:
                    ll_max_nonneg = ll

            if ll_max is None:
                ll_max = ll
                nchosen_max = nchosen
            elif ll > ll_max:
                ll_second = ll_max
                ll_max = ll
                nchosen_max = nchosen
            elif ll == ll_max:
                # This will cause LLR = 0
                ll_second = ll_max
            else:
                if ll_second is None or ll > ll_second:
                    ll_second = ll
                break
        
        ll_ratio = 0
        if ll_max is not None and ll_second is not None:
            ll_ratio = ll_max - ll_second
        name = 'Negative'
        if nchosen_max == 1:
            name = str(vals.index[0])
        elif nchosen_max <= max_cells:
            assignments_cell = []
            for i in range(0, nchosen_max):
                assignments_cell.append(str(vals.index[i]))
            name = "+".join(sorted(assignments_cell))
        else:
            # Let the LLR for "negative" assignments be the max LL minus the 
            # max LL of a non-negative assignment.
            ll_ratio = ll_max - ll_max_nonneg

        assignments.append([name, nchosen_max, ll_max, ll_ratio])
        
        if (idx + 1) % 1000 == 0:
            print("Processed {} barcodes".format(idx + 1), end="\r", file=sys.stderr)

    print("", end="\n", file=sys.stderr)

    assignments = pd.DataFrame(assignments, columns=['name', 'n_indv', 'll', 'llr'])
    assignments = assignments.astype({'n_indv': int})
    # Replace num individuals with 0 for those inferred to be noise
    assignments.loc[assignments['n_indv'] > max_cells,'n_indv'] = 0
    #assignments.drop(['n_indv'], axis=1, inplace=True)
    assignments.index = counts.index
    
    # Drop assignments where llr == 0 (not a real decision)
    assignments = assignments.loc[assignments['llr'] > 0,:]

    assignments.to_csv(sys.stdout, sep='\t', header=False)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

