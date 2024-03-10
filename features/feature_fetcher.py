from . import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
from . import fingerprint_2d as fingerprint
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
import pandas as pd
import numpy as np
import sys
import multiprocessing
from functools import partial
import torch 


class FeatureFetcher():
  def __init__(self,cfg,tokenizer):

    self.cfg = cfg.features
    self.max_length = cfg.data.max_length
    self.task = cfg.task.type
    self.tokenizer = tokenizer
    # self.max_length = cfg.data.max_length
    
  def tokenize_function(self,examples,add_special_tokens):
      return self.tokenizer(examples,max_length=self.max_length, padding="max_length", truncation=True,add_special_tokens=add_special_tokens)
  
  '''
  for hand-crafted features, derive from AMPSpace
  '''
  def query_single_hc(self,peptide):
      peptide = str(peptide)
            
      AAC = list(AAComposition.CalculateAAComposition(peptide).values())
      DIP = list(AAComposition.CalculateDipeptideComposition(peptide).values())
      MBA = list(Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5).values())
      CCTD = list(CTD.CalculateCTD(peptide).values())
      QSO = list(QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5).values())
      PAAC = list(PseudoAAC._GetPseudoAAC(peptide,lamda=5).values())
      APAAC = list(PseudoAAC.GetAPseudoAAC(peptide, lamda=5).values())
      Basic = list(BasicDes.cal_discriptors(peptide).values())
      
      desc_alpha = PeptideDescriptor(peptide, 'levitt_alpha')
      desc_alpha.calculate_global()  # 计算alpha-螺旋倾向性
      helix = list(desc_alpha.descriptor.reshape(-1))
      
      
      return AAC + DIP + MBA + CCTD + QSO + PAAC + APAAC + Basic + helix
  '''
  for fingerprint feature derive from peptimizer
  '''

  def query_single_fingerprint(self, sequence):
      fp_seq = self.fp.seq(sequence)
      n_rows = self.max_length - len(sequence)
      shape_padding = (n_rows, 2048)
      padding_array = np.zeros(shape_padding)
      fp_seq = np.concatenate((fp_seq, padding_array), axis = 0)
      return fp_seq
  
  def query_features(self, peptides):
      
      if self.cfg.type == 'AMPSpace':
      
        peptides_descriptors = {'x':self.build_index(peptides,self.get_dict()),
                                 'length': [len(_) for _ in peptides]}

      if (self.cfg.type == 'CellTree-rnn') or (self.cfg.type == 'CellTree-cnn') or (self.cfg.type == 'SeqUNet'):
        
        peptides_descriptors = {'x':self.prepare_Reg(peptides)}
        
      if self.cfg.type == 'peptimizer':
        self.fp = fingerprint.Fingerprint_Generation(smiles_file = "/data/home/scv6872/Breeze/features/cpp_smiles.json", 
                                               nbits = 2048, radius = 3)

        peptides_descriptors = {'x':[self.query_single_fingerprint(peptide) for peptide in peptides]}
    
      if self.cfg.type == 'HC':
      
        peptides_descriptors = {'x':[]}
        for peptide in peptides:
  
          peptides_descriptors['x'].append(self.query_single_hc(peptide))
        
        
      
      if self.cfg.type == 'breeze':
        
        if self.task == 'pretrain':
          tokenize_function_partial = partial(self.tokenize_function, add_special_tokens=False)
        else:
          tokenize_function_partial = partial(self.tokenize_function, add_special_tokens=True)
        
        peptides_descriptors = tokenize_function_partial(peptides)
        
      return peptides_descriptors
  

  '''
  for AMPSpace
  '''
  def get_dict(self):
      Letter_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                     'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
      return Letter_dict
  
  
  def get_reverse_dict(self):
      reverse_dict = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
                      13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'}
      return reverse_dict
  
  
  def build_index(self,data, Letter_dict):
      """
      Preprocess
      负责词表的映射
      建立词向量
      input: data 序列
      output: 映射完的数据
      """
      data_process = []
      for i in range(len(data)):
          tmp = []
          
          for j in range(len(data[i])):
              tmp.append(Letter_dict[data[i][j]])
          if len(data[i]) < self.max_length:                          
              npi = np.zeros((self.max_length - len(data[i])), dtype=np.int)
              tmp.extend(npi)  
          data_process.append(tmp)
      
          
      return data_process
      
  '''
  for CellTree
  '''
  def onehot_encoding(self,data):
    
    """One-hot encoding of DNA or protein sequences

    Args:
        data (list): List of sequence strings to encode dim:(N, sequence length)
        alphabet (string, optional): The alphabet to use; either DNA or Amino Acid. Defaults to AAalphabet.

    Returns:
        list: List of encoded sequences dim:(N, sequence length, alphabet length)
    """
    alphabet = 'BCDSQKIPTFNGHLRWAVEYM-'
    aa2hot = {}
    for i, aa in enumerate(alphabet):
        v = [0 for j in alphabet]
        v[i] = 1
        aa2hot[aa] = v

    onehot_encoded = []
    for seq in data:
        temp = []
        for aa in seq:
            temp.append(aa2hot[aa])
        onehot_encoded.append(temp)
    return onehot_encoded
    
    
  def padding(self,data, begin_token='', end_token='-', lim=48):
    """Pads all sequences in the list to a certain length with an end token

    Args:
        data (list): List of sequences as strings 
        begin_token (str, optional): Character to pad the beginning of each sequence string. Defaults to ''.
        end_token (str, optional): Character to pad the end of each sequence string to reach the length limit. Defaults to '-'.
        lim (int, optional): Length thereshold. Defaults to 48.

    Returns:
        list: List of padded sequences 
    """
    padded = []
    for seq in data:
        temp = begin_token + seq + end_token * (lim - len(seq))
        padded.append(temp)

    return padded
    
    
  def prepare_Reg(self,data):
    
    seq = data
    seq = self.padding(seq,lim=self.max_length)
    seq = self.onehot_encoding(seq)
    # seq = torch.tensor(seq)
    return seq
'''
if __name__ == "__main__":

    FeatureFetcher(cfg)
    tmp = cal_pep("RIWVIWRR").values
'''