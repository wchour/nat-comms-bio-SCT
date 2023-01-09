# Created by: William Chour
# Last updated: January 9, 2023
#
# This script takes in a csv list of peptide sequences, and generates a csv file
# containing corresponding peptide-encoded primers for SCT generation. The
# primers are generated according to NUPACK optimization parameters, and thus
# will require the user to install NUPACK prior to running this script.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import Seq
from Bio.Seq import Seq
import itertools
import nupack.utils as npk
import time

# Dictionary containing sense codons for amino acids. For each aa, first list
# contains cumulative probability of each codon occurence. Second list contains
# all possible codons for each aa. Third list contains cumulative probability
# only for codons which occur at probability of 0.13 or greater. Fourth list
# is the corresponding filtered codon.
sense_5to3_stats = {
    'R': ([.08, .26, .37, .57, .78, 1],
          ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg'],
          [.225, .475, .7375, 1],
          ['cgc', 'cgg', 'aga', 'agg']),
    'H': ([.42, 1],
          ['cat', 'cac'],
          [.42, 1],
          ['cat', 'cac']),
    'K': ([.43, 1],
          ['aaa', 'aag'],
          [.43, 1],
          ['aaa', 'aag']),
    'D': ([.46, 1],
          ['gat', 'gac'],
          [.46, 1],
          ['gat', 'gac']),
    'E': ([.42, 1],
          ['gaa', 'gag'],
          [.42, 1],
          ['gaa', 'gag']),
    'S': ([.19, .41, .56, .61, .76, 1],
          ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc'],
          [.2, .431579, .589474, .747368, 1],
          ['tct', 'tcc', 'tca', 'agt', 'agc']),
    'T': ([.25, .61, .89, 1],
          ['act', 'acc', 'aca', 'acg'],
          [.280899, .685393, 1],
          ['act', 'acc', 'aca']),
    'N': ([.47, 1],
          ['aat', 'aac'],
          [.47, 1],
          ['aat', 'aac']),
    'Q': ([.27, 1],
          ['caa', 'cag'],
          [.27, 1],
          ['caa', 'cag']),
    'C': ([.46, 1],
          ['tgt', 'tgc'],
          [.46, 1],
          ['tgt', 'tgc']),
    'G': ([.16, .5, .75, 1],
          ['ggt', 'ggc', 'gga', 'ggg'],
          [.16, .5, .75, 1],
          ['ggt', 'ggc', 'gga', 'ggg']),
    'P': ([.29, .61, .89, 1],
          ['cct', 'ccc', 'cca', 'ccg'],
          [.325843, .685393, 1],
          ['cct', 'ccc', 'cca']),
    'A': ([.27, .67, .9, 1],
          ['gct', 'gcc', 'gca', 'gcg'],
          [.3, .74444, 1],
          ['gct', 'gcc', 'gca']),
    'I': ([.36, .83, 1],
          ['att', 'atc', 'ata'],
          [.36, .83, 1],
          ['att', 'atc', 'ata']),
    'L': ([.08, .21, .34, .54, .61, 1],
          ['tta', 'ttg', 'ctt', 'ctc', 'cta', 'ctg'],
          [.151163, .302326, .534884, 1],
          ['ttg', 'ctt', 'ctc', 'ctg']),
    'M': ([1],
          ['atg'],
          [1],
          ['atg']),
    'F': ([.51, 1],
          ['ttt', 'ttc'],
          [.51, 1],
          ['ttt', 'ttc']),
    'W': ([1],
          ['tgg'],
          [1],
          ['tgg']),
    'Y': ([.52, 1],
          ['tat', 'tac'],
          [.52, 1],
          ['tat', 'tac']),
    'V': ([.18, .42, .54, 1],
          ['gtt', 'gtc', 'gta', 'gtg'],
          [.204545, .477273, 1],
          ['gtt', 'gtc', 'gtg'])}

def ensemble_defect(sequence):
    """
    Calculates ensemble defect of a sequence, returning the defect value and stop condition status.
    """
    # Take in sequence.
    seq = str(sequence)

    # Find length of sequence.
    N = len(seq)

    # Construct pairlist of sequence at desired temperature.
    P = npk.pairs(seq, T=23, NUPACKHOME="/usr/local")

    # Use 'fake_seq' to represent a linearized DNA strand of length N.
    fake_seq = 'A'*N

    # Generate NPK structure representation and pairlist of our linear 'fake_seq'.
    structure, _ = npk.mfe(fake_seq, T=23, NUPACKHOME="/usr/local")
    pairlist, _ = npk.dotparens_2_pairlist(structure)
    S = np.zeros([N, N+1])

    # Compute defect energy.
    for i in range(len(pairlist)):
        a, b = pairlist[i]
        S[a][b] = 1
        S[b][a] = 1
    for i in range(N):
        if np.sum(S[i][0:N+1]) == 0:
            S[i][N] += 1

    sigma = 0
    for i in range(N):
        for j in range(N+1):
            sigma += S[i][j] * P[i][j]

    stop_condition = N/100
    defect = N - sigma
    stop_status = defect <= stop_condition

    # Return mfe defect and whether or not stop condition is satisfied
    return defect, stop_status

def codon_picker(amino_acid):
    '''
    For a given amino acid, uses a reference table of degenerate codons + probabilities to
    select one based on RNG value.
    '''

    # Instantiate index for codon table.
    picked_index = 0

    # Identify correct list of codons for amino acid.
    reference = sense_5to3_stats[amino_acid]

    # Randomize variable x.
    x = np.random.random()

    # Identify which codon to pick for x given probabilities of each codon.
    for i in range(len(reference[2])-1):
        higher = len(reference[2]) - 1 - i
        lower = len(reference[2]) - 2 - i
        if reference[2][lower] < x < reference[2][higher]:
            picked_index = higher

    # Select the right codon as output.
    picked_codon = reference[3][picked_index]

    return picked_codon

def initial_peptide_dna(peptide_aa_seq):
    '''
    Initialize DNA representation of a peptide sequence in 5to3 sense format.
    '''

    # Instantiate DNA sequence variable.
    peptide_dna_seq = ''

    # For given peptide sequence, pick a codon per amino acid using codon_picker function.
    for i in range(len(peptide_aa_seq)):
        peptide_dna_seq += (codon_picker(peptide_aa_seq[i]))

    return peptide_dna_seq

def bind_peptide_combiner(primer_bind_5to3, peptide_dna_seq_sense_5to3, direction):
    '''
    Given primer binding site, 5to3 sense DNA sequence of peptide, and direction of primer,
    generates a full primer.
    '''
    # Reverse direction primer requires primer binding site in front of peptide DNA,
    # followed by reverse complement function.
    if direction == 'rev':
        initial_neo_primer = Seq(primer_bind_5to3 + peptide_dna_seq_sense_5to3)
        initial_neo_primer = initial_neo_primer.reverse_complement()

    # Forward direction primer simply requires primer binding site after peptide DNA.
    if direction == 'fwd':
        initial_neo_primer = Seq(peptide_dna_seq_sense_5to3 + primer_bind_5to3)

    return initial_neo_primer

def position_picker(peptide):
    '''
    Given a peptide, randomly select an amino acid position for subsequent
    mutation (by mutate_peptide function). Avoid amino acids without degeneracy (W & M).
    '''

    # Initialize variable to indicate True when substitutable position has been found.
    status = False

    # While position has not been found...
    while status == False:

        # Randomly pick a position and check if it has degeneracy. If true,
        # then process is finished.
        position = np.random.randint(low = 0, high = len(peptide))
        if peptide[position] == 'W' or peptide[position] == 'M':
            status = False
        else:
            status = True

    return position

def mutate_peptide(peptide_dna_seq_sense_5to3, peptide):
    '''
    Given a peptide sequence and its corresponding 5to3 sense DNA,
    mutates one position into a degenerate codon.
    '''

    # Call on position_picker to pick a position for mutation.
    position = position_picker(peptide)

    # Record the original codon at picked position.
    old_codon = peptide_dna_seq_sense_5to3[position*3:position*3+3]

    # Assign new codon variable to be identical to old codon.
    new_codon = old_codon

    # As long as the new codon has not been updated, pick a new degenerate codon.
    while old_codon == new_codon:
        new_codon = codon_picker(peptide[position])

    # Once new codon has been selected, update peptide DNA sequence with new codon.
    peptide_dna_seq_sense_5to3 = peptide_dna_seq_sense_5to3[:position*3] + new_codon + peptide_dna_seq_sense_5to3[(position*3 + 3):]

    return peptide_dna_seq_sense_5to3

def primer_optimize(peptide_aa_seq):
    '''
    Given a peptide sequence, optimize primer according to ensemble defect. Output is
    defect value and primer sequence.
    '''

    # Initiate timer to record optimization time.
    start = time.time()

    # Initialize the DNA sequence of peptide.
    peptide_dna_seq = initial_peptide_dna(peptide_aa_seq)

    # Produce peptide primer, accounting for bind site and directionality.
    peptide_primer = bind_peptide_combiner(primer_bind_5to3, peptide_dna_seq, direction)

    # Initialize min defect ceiling and desired iterations.
    min_defect = 100
    count = 1000

    # Initialize variables to hold temporary test primer and best primer found.
    test_primer = peptide_primer
    best_primer = peptide_primer

    # For each iteration...
    for i in range(count):

        # Calculate defect of best primer.
        defect_result, _ = ensemble_defect(test_primer)

        # If defect is lower than before, save new defect as threshold and record primer as best.
        if defect_result < min_defect:
            min_defect = defect_result
            best_primer = peptide_primer

        # Regardless of defect result, make a new peptide DNA sequence and attach it to primer.
        peptide_dna_seq = mutate_peptide(peptide_dna_seq, peptide_aa_seq)
        peptide_primer = bind_peptide_combiner(primer_bind_5to3, peptide_dna_seq, direction)

        # Save the new primer as the test primer for iteration.
        test_primer = peptide_primer

    # Record elapsed time.
    end = time.time()
    elapsed_seconds = end - start
    print('Optimized ' + peptide_aa_seq + ' primer in %4.2f seconds.' %(elapsed_seconds))
    print('Primer: ' + best_primer)
    print('Defect: %4.2f' %(min_defect))

    return min_defect, best_primer

def nusub_fast(primer_bind_5to3, direction, peptides):
    '''
    Given primer bind site, primer direction, and peptide list,
    generate list of optimized linear primers.
    '''

    # Instantiate lists to hold selected primers,
    # defect values, and the peptide DNA sequence for each peptide.
    primers = []
    defects = []
    antigen_seq = []

    # For each peptide, generate optimized primer and save relevant information to lists.
    for i in range(len(peptides)):
        peptide = peptides[i].replace(u'\xa0', u'').replace(' ', '')
        defect, best_primer = primer_optimize(peptide)
        if direction == 'rev':
            rev_comp_antigen = ''.join(ch for ch in best_primer if not ch.isupper())
            sense_antigen = Seq(rev_comp_antigen).reverse_complement()
        if direction == 'fwd':
            sense_antigen = ''.join(ch for ch in best_primer if not ch.isupper())
        primers.append(str(best_primer))
        defects.append(defect)
        antigen_seq.append(str(sense_antigen))

    # Zip together lists.
    data_tuples = list(zip(peptides, primers, defects, antigen_seq))

    # Save data into dataframe. Re-index dataframe to begin with 1.
    df_out = pd.DataFrame(data_tuples, columns=['peptide','primer', 'defect', 'antigen_seq'])
    df_out.index = np.arange(1, len(df_out)+1)

    return df_out

# Gather all relevant information prior to optimizing primers.
# primer_bind_5to3 example: GAGCAGCTGTTCTGTTGGC
primer_bind_5to3 = input('What is the 5-to-3 DNA sequence of your primer binding site? ')
direction = input('What is the direction of your primer? [rev/fwd]')
peptide_list = input('Enter peptides file name (w/o .csv): ');
file_name = input('What is the name of output file?: ')
df = pd.read_csv(peptide_list + '.csv', header = 0)
column_name = df.columns[0]
peptides = df[column_name].tolist()

# Save optimizer primer list to csv file.
output = nusub_fast(primer_bind_5to3, direction, peptides)
output.to_csv(file_name + '.csv')
