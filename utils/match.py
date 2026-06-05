import math
import time
from typing import Tuple, Dict, List
import os
import sys
import math
import numpy as np
import pandas as pd
from AMPCliff.features import fingerprint_2d as fingerprint
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
from rdkit import DataStructs
from itertools import combinations
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm 
import ipdb
from Bio import pairwise2
from Bio.Align import substitution_matrices

def smith_waterman(seq1, seq2, scoring_matrix=substitution_matrices.load("BLOSUM62"), gap_open=-11, gap_extend=-1, lambda_value=0.251, K=0.031):
# def smith_waterman(seq1, seq2, scoring_matrix=matlist.blosum62, gap_open=-10, gap_extend=-0.5, lambda_value=0.251, K=0.031):

    
    # Smith-Waterman for alignment
    alignments = pairwise2.align.localds(seq1, seq2, scoring_matrix, gap_open, gap_extend)
    
    if len(alignments): 
      # select the highest score result
      top_alignment = alignments[0]
    
      aligned_seq1, aligned_seq2, score, start, end = top_alignment
      
      # calculate the number of residues match the same position
      identical_residues = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
      aligned_columns = max(len(aligned_seq1), len(aligned_seq2))  # the number of aligned columns(include gaps)
      
      # calculate bit score using the new lambda and K values
      bit_score = (lambda_value * score - math.log(K)) / math.log(2)
      
      # calculate the index needed
      identity_ratio = identical_residues / aligned_columns
      
      # calculate bit score per column
      bit_score_per_column = bit_score / aligned_columns if aligned_columns else 0
      
      # print result
      print(f"Aligned Seq 1: {aligned_seq1}")
      print(f"Aligned Seq 2: {aligned_seq2}")
      print(f"Score: {score}")
      print(f"Identity Ratio: {identity_ratio:.4f}")
    
    else:
      identity_ratio = 0
      score = 0
      aligned_seq1, aligned_seq2 = None, None
      bit_score_per_column = 0
    # return results
    return identity_ratio, aligned_seq1, aligned_seq2, score, bit_score_per_column
    

def calculate_average_blosum62(seq1, seq2,tanimoto_matrix):
    """
    For two aligned sequences' two-dimensional fingerprint arrays, calculate the Tanimoto similarity between each pair of amino acid fingerprints, and then take the average value as the overall similarity between the sequences.
    
    parameters:
    - seq1: aligned first sequence, shape in [aligned length].
    - seq2: aligned second sequence, shape in [aligned length].
    - tanimoto_matrix: similarity among each 2 amino acids and gap
    
    return:
    - average_similarity: the average tanimoto simularity of all the position of a sequence pair
    - average_dissimilarity: the average tanimoto simularity of the position in difference of a sequence pair
    """
    total_similarity = 0
    dissimilarity = 0
    dis_pairs = 0
    
    num_pairs = len(seq1)  # the number of pairs
    
    for i in range(num_pairs):
    
        similarity = tanimoto_matrix.loc[seq1[i],seq2[i]]
        total_similarity += similarity
        
        if seq1[i] != seq2[i]:
            dissimilarity += similarity
            dis_pairs += 1
    
    average_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
    average_dissimilarity = dissimilarity / dis_pairs if dis_pairs > 0 else 0
    
    return average_similarity, average_dissimilarity

    
    
def Blosum62(seq1, seq2):
    
    normalized_blosum62 = pd.read_csv("/data/home/scv6872/AMPCliff/utils/blosum62_normalized.csv",index_col=0)
    # ipdb.set_trace()
    average_similarity, average_dissimilarity = calculate_average_blosum62(seq1, seq2,normalized_blosum62)
    
    return average_similarity, average_dissimilarity
    
    
def levenstein_align(seq1, seq2):
    
    differences = sum(1 for base1, base2 in zip(seq1, seq2) if base1 != base2)
    
    mutations = []
    for pos, (base1, base2) in enumerate(zip(seq1, seq2), start=1):  
        if base1 != base2:
            mutations.append(f"({pos}, {base1}, {base2})")
    
    mutation_info = '|'.join(mutations)
    return differences, mutation_info
  

def calculate_average_similarity(seq1, seq2,tanimoto_matrix):
    """
    For two aligned sequences' two-dimensional fingerprint arrays, calculate the Tanimoto similarity between each pair of amino acid fingerprints, and then take the average value as the overall similarity between the sequences.
    
    parameters:
    - seq1: aligned first sequence, shape in [aligned length].
    - seq2: aligned second sequence, shape in [aligned length].
    - tanimoto_matrix: similarity among each 2 amino acids and gap
    
    return:
    - average_similarity: the average tanimoto simularity of all the position of a sequence pair
    - average_dissimilarity: the average tanimoto simularity of the position in difference of a sequence pair
    """
    total_similarity = 0
    dissimilarity = 0
    dis_pairs = 0
    
    num_pairs = len(seq1)  # the number of pairs
    
    for i in range(num_pairs):
    
        similarity = tanimoto_matrix.loc[seq1[i],seq2[i]]
        total_similarity += similarity
        
        if seq1[i] != seq2[i]:
            dissimilarity += similarity
            dis_pairs += 1
    
    average_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
    average_dissimilarity = dissimilarity / dis_pairs if dis_pairs > 0 else 0
    
    return average_similarity, average_dissimilarity

    
    
def TanimotoSimilarity(seq1, seq2):
    
    tanimoto_matrix = pd.read_csv("/data/home/scv6872/AMPCliff/utils/tanimoto.csv",index_col=0)
    
    # ipdb.set_trace()
    average_similarity, average_dissimilarity = calculate_average_similarity(seq1, seq2,tanimoto_matrix)
    
    return average_similarity, average_dissimilarity


def process_sequence_pair(seq_pair,diff):
    idx1, seq1 = seq_pair[0]
    idx2, seq2 = seq_pair[1]
    
     # calculate the subtraction of activities
    dmic = seq1['Activity'] - seq2['Activity']
    activity_diff = np.abs(dmic)
    
    is_ac = False
    if dmic > np.log10(diff):
      lev = levenshtein_distance(seq1['Sequence'], seq2['Sequence'])
      
      seq_id, align1, align2, score, bit_score_per_column = smith_waterman(seq1['Sequence'], seq2['Sequence'])
      # '''
      if seq_id:
          
      # tanimoto average, blosum62 average, levenstein align, mutation
          sim,dissim = TanimotoSimilarity(align1, align2)
          
          avg_sim, avg_dis = Blosum62(align1, align2)
          
          leven_dist, mutation_info = levenstein_align(align1, align2)
          
          if avg_sim >= 0.9 or sim >= 0.9:
            is_ac = True
         
      else:
          sim = 0
          dissim = 0
          avg_sim, avg_dis = 0, 0
          leven_dist, mutation_info = None, None
      # '''
     
      
      # add activity cliff pairs
      sequence1,mic1 = seq1['Sequence'],seq1['Activity'] 
      sequence2,mic2 = seq2['Sequence'],seq2['Activity']
      
      # save pred results
      pred_mic1,pred_mic2 = seq1['pred'],seq2['pred']
      pred_dmic = pred_mic1-pred_mic2
      
      acindex = pred_dmic/dmic-1 # closer to 0 is better
      
    if is_ac: 
      # return ac result
      return {
          'seq1': sequence1,
          'seq2': sequence2,
          'mic1': mic1,
          'pred_mic1':pred_mic1,
          'mic2': mic2,
          'pred_mic2':pred_mic2,
          'dmic': dmic, 
          'pred_dmic':pred_dmic,
          'ACIndex':acindex,
          'alignment seq1': align1,
          'alignment seq2': align2,
          'lev':lev,
          'seq id':seq_id,
          'score': score,
          'bit score': bit_score_per_column,
          'tanimoto simularity':sim,
          'tanimoto dissimularity':dissim,
          'blosum62 average':avg_sim,
          'blosum62 disaverage':avg_dis,
          'levenstein aligned':leven_dist,
          'mutation':mutation_info
      }
    else:
       return None
    
def seq2pair(testset,diff):
   
    # create sequence pairs
    sequence_pairs = list(combinations(testset.iterrows(), 2))
    
    new_rows = []
    
    for pair in sequence_pairs:
      new_row = process_sequence_pair(pair,diff)
      if new_row is not None:
        new_rows.append(new_row)
    
    # save result
    ac_sequences = pd.DataFrame(new_rows).drop_duplicates(subset=None, keep='first', inplace=False)
    return ac_sequences

if __name__=='__main__':
    main()