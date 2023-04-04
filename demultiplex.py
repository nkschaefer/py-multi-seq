#! /usr/bin/env python3
import re
import fuzzywuzzy
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import sys
import os
import tqdm
import numpy as np
import sparse_dot_topn
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from scipy import io
from scipy.sparse import csr_matrix
import argparse
import gzip
import subprocess
"""
Python implementation of MULTIseq barcode alignment using fuzzy string matching.
Takes FASTQs, along with information matching MULTIseq barcodes to sample-specific identifiers
and outputs a counts table to the desired output directory.
Also outputs data in sparse matrix format, for loading into tools like scanpy.
"""

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", "-1", nargs="+", \
        help="R1 FASTQ file(s) for a single batch. Can be gzipped. R1 and R2 must be in same order.", \
        required=True)
    parser.add_argument("--r2", "-2", nargs="+", \
        help="R2 FASTQ file(s) for a single batch. Can be gzipped. R1 and R2 must be in same order.", \
        required=True)
    parser.add_argument("--key", "-k", required=True, \
        help="File mapping multiseq indices to sample names. Should be tab-separated, first column = identifier, \
second column = MULTIseq well ID. Well ID must match values in MultiseqIndices file.")
    parser.add_argument("--barcodes", "-b", required=True, \
        help="Path to (possibly gzipped) valid cell barcodes, i.e. for cellranger runs, \
this is <output directory>/outs/filtered_feature_bc_matrix/barcodes.tsv.gz")
    parser.add_argument("--indices", "-i", required=False, \
        default=os.path.dirname(os.path.realpath(__file__)) + "/MultiseqIndices.txt", \
        help="Path to multiseq indices (default = use included multiseq indices)")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory for output files. \
Must not collide with output_dir for another run, or files will be overwritten.")
    parser.add_argument("--barcode_len", "-bl", type=int, required=False, default=16, \
        help="Cell barcode length (default = 16, for 10X Genomics)")
    parser.add_argument("--index_len", "-il", type=int, required=False, default=8, \
        help="Length of the R2 index (default = 8, for MULTIseq)")
    parser.add_argument("--mismatch_tolerance", "-m", type=int, required=False, default=1, \
        help="Allowable number of mismatches in barcode (default = 1)")
    parser.add_argument("--flipped", "-f", action="store_true", \
        help="Specify if you used the multiseq index plate in the wrong orientation (oops)")
    return parser.parse_args()

"""
Read in Fastq information
Adapted from: https://www.biostars.org/p/317524/
Function to parse fastq
"""
def processfq(lines=None):
    ks = ['name', 'sequence', 'optional', 'quality']
    return {k: v for k, v in zip(ks, lines)}

"""
Function to read through paired FASTQ files (gzipped or not) and store
relevant chunks of sequences in pandas DataFrame.
"""
def read_fq(reads1, reads2, f_len, r_len):
    
    n = 4
    readpairs=[]
    i=0
    
    for idx1 in range(0, len(reads1)):
        if idx1 > len(reads2)-1:
            print("ERROR: unequal number of forward and reverse FASTQ files", file=sys.stderr)
            exit(1)
        
        # Open both files in pair
        fname1 = reads1[idx1]
        fname2 = reads2[idx1]

        f1 = None
        f2 = None

        f1_gz = False
        f2_gz = False

        if fname1[-3:] == '.gz':
            f1 = gzip.open(fname1, 'r')
            f1_gz = True
        else:
            f1 = open(fname1, 'r')
        if fname2[-3:] == '.gz':
            f2 = gzip.open(fname2, 'r')
            f2_gz = True
        else:
            f2 = open(fname2, 'r')

        # Iterate through all reads in both files
        flines = []
        rlines = []
        linef = f1.readline()
        liner = f2.readline()
        
        line_idx = 0
        while linef and liner:
            if f1_gz:
                flines.append(linef.rstrip().decode())
            else:
                flines.append(linef.rstrip())
            if f2_gz:
                rlines.append(liner.rstrip().decode())
            else:
                rlines.append(liner.rstrip())
            if (len(flines) == n) and (len(rlines) == n):
                recordf = processfq(flines)
                recordr = processfq(rlines)
                
                if recordf['name'][1:].split()[0] != recordr['name'][1:].split()[0]:
                    print("ERROR: sequence mismatch in {} and {}, line {}: {} != {}".format( \
                        fname1, fname2, line_idx, \
                        recordf['name'][1:].split()[0], recordr['name'][1:].split()[0]), file=sys.stderr)
                    exit(1)
                
                readpairs.append([recordf['sequence'][0:int(f_len)],recordr['sequence'][0:int(r_len)]])
                flines = []
                rlines = []
            
            i+=1
            if i%100000==0:
                print("Processed {} seqs".format(i), end="\r", file=sys.stderr)

            linef = f1.readline()
            liner = f2.readline()
            line_idx += 1
        
        f1.close()
        f2.close()

    print("", end="\n", file=sys.stderr)
    
    df=pd.DataFrame(readpairs)
    return df

def ngrams(string, n=8):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, name_vectorx,name_vectory, top=100):
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    sparserows=sparse_matrix.argmax(1).A1
    sparsecols=list(range(len(name_vectory)))
    left_side = name_vectorx
    right_side = np.array(name_vectory)[sparserows]
    similarity=[]
    for x,y in zip(list(range(len(name_vectorx))),sparserows):
        similarity.append(sparse_matrix[x,y])
    
    return pd.DataFrame({'original': left_side, \
        'matched': right_side, \
        'similarity': similarity})


def main(args):
    options = parse_args()
    
    # Read MULTIseq indices
    Indices = pd.read_csv(options.indices, sep='\t', header=0)
    Indices.index=Indices.index+1
    cols = ['Well Position', 'Name', 'Barcode_Sequence', 'Oligo_sequence']
    for col in cols:
        if col not in Indices.columns:
            print("ERROR: column {} not found in index file".\
                format(col), file=sys.stderr)
            exit(1)

    # Determine type of "key" file
    # Options are either UID -> multiseq well mappings OR
    # a file with columns Sample and Multiseq_Index, 
    # where Sample == UID and Multiseq_Index = numeric, starting from 1
    key = None
    
    # Peek at file
    f = open(options.key, 'r')
    file_wells = False
    file_indices = False
    for line in f:
        line = line.rstrip()
        col1, col2 = line.split('\t')
        if col1 == "Sample" and col2 == "Multiseq_Index":
            file_indices = True
        elif col2 in list(Indices['Well Position']):
            file_wells = True
        break
    
    if file_wells:
        # Read UID -> multiseq well mappings
        key = pd.read_csv(options.key, sep='\t', names=['Sample', 'Well'])
        key = key.replace(np.nan, '', regex=True)
        
        # Merge with Indices file to convert well to (1-based) index in Index file
        Indices2 = Indices
        Indices2['idx'] = Indices2.index
        key = key.merge(Indices2, left_on='Well', right_on='Well Position')
        key.drop(['Well', 'Well Position', 'Name', 'Barcode_Sequence', "Oligo_sequence"], axis=1, inplace=True)
        key.rename({'idx': "Multiseq_Index"}, axis=1, inplace=True)
        key.index = key['Multiseq_Index']

    elif file_indices:
        # Just load file; already formatted as needed
        key = pd.read_csv(options.key, sep='\t', header=0)
    else:
        print("ERROR: input file {} does not appear to be either \
            sample -> multiseq well ID mappings or sample -> multiseq index \
            number (1-based) mappings, with header row Sample<tab>Multiseq_Index".format(options.key), \
            file=sys.stderr)
        exit(1)

    cellfile = options.barcodes

    outfile = options.output_dir

    # length of forward read to use
    f_len = options.barcode_len
    # length of reverse read to use
    r_len = options.index_len
    
    # Find the directory containing this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    if options.flipped:
        eighttwelve=pd.read_csv('{}/8-12IndexConversion.txt'.format(script_dir), sep='\t')
        etdict=dict(zip(eighttwelve['8-Index'],eighttwelve['12-Index']))
        key['Multiseq_Index']=[etdict[x] for x in key['Multiseq_Index']]
    
    # Somewhat adhoc for my multiseq format key
    keydict=dict(zip(key['Multiseq_Index'],key['Sample']))

    df = read_fq(options.r1, options.r2, f_len, r_len)
    cells=list(pd.read_csv(cellfile, header=None, index_col=False, sep='\t')[0])
    
    cellset=list(set(cells))
    Indices=Indices.loc[Indices.index.isin(key.index),:]
    inds=list(Indices['Barcode_Sequence'])
    inddictrev=dict(zip(Indices.index,Indices['Barcode_Sequence']))
    inddict=dict(zip(inddictrev.values(),inddictrev.keys()))

    celldictrev=dict(enumerate(cellset))
    celldict=dict(zip(celldictrev.values(),celldictrev.keys()))

    mat=np.zeros((len(cellset),len(inds)),dtype=np.int32)

    bcfixer=[process.extractOne(x,inddictrev) for x in tqdm.tqdm(df[1].unique())]
    bcfixerdict=dict(zip(df[1].unique(),bcfixer))
    bcfixed=[bcfixerdict[x] for x in df[1]]

    df=df.loc[[x is not None for x in bcfixed],:]
    df[1]=[x[2] for x in bcfixed if x is not None]
    #print(df,flush=True, file=sys.stderr)

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(cellset+list(df[0]))

    tf_idf_matrix_ref=tf_idf_matrix[:len(cellset),:]

    tf_idf_matrix_query=tf_idf_matrix[len(cellset):,:]
    matches = awesome_cossim_top(tf_idf_matrix_query,tf_idf_matrix_ref.transpose(), 1)

    matchdf=get_matches_df(matches,list(df[0]),cellset,False)

    df[0]=matchdf['matched']

    df=df.loc[list(matchdf['similarity']>.7),:]
    
    # Write out counts to file
    mat=df.groupby([0,1]).size().unstack(fill_value=0)
    mat.columns=[keydict[x] for x in mat.columns]
    mat.astype('int')
    mat.index.rename('barcode', inplace=True)
    mat.to_csv(os.path.join(outfile,'MULTIseq_counts.txt'),sep='\t',header=True,index=True)
    
    # Write out counts to files in market exchange format
    if not os.path.exists(os.path.join(outfile,'multiseq_outs')):
        os.mkdir(os.path.join(outfile,'multiseq_outs'))

    scipy.io.mmwrite(os.path.join(outfile,'multiseq_outs','matrix.mtx'),scipy.sparse.csr_matrix(np.array(mat).T))
    with open(os.path.join(outfile,'multiseq_outs','barcodes.tsv'),'w') as f:
        for ind,i in enumerate(mat.index):
            f.write(i)
            f.write("\n")
            
    with open(os.path.join(outfile,'multiseq_outs','features.tsv'),'w') as f:
        for ind,i in enumerate(mat.columns):
            f.write(str(ind)+'\t'+str(i)+'\tAntibody Capture')
            f.write("\n")
    
    # Assign barcodes to identities using counts file just written
    print("Assign cell barcodes to most likely identity...", file=sys.stderr)

    assignments_out = os.path.join(outfile, 'MULTIseq_assignments.txt')
    assignments_out_f = open(assignments_out, 'w')
    p = subprocess.Popen([os.path.join(script_dir, 'assign_barcodes.py'), \
        os.path.join(outfile, 'MULTIseq_counts.txt')], stdout=assignments_out_f)
    out, err = p.communicate()
    assignments_out_f.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
