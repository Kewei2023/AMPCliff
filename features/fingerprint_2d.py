# maintained by kewei li
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

class Fingerprint_Generation:
    def __init__(self, smiles_file, nbits, radius):
        self.nbits = nbits
        self.radius = radius
        self.smiles_to_mol = self._load_smiles(smiles_file)

    def _load_smiles(self, smiles_file):
        """从JSON文件加载SMILES字符串并转换为RDKit分子对象的字典"""
        with open(smiles_file, 'r') as f:
            smiles_dict = json.load(f)
        smiles_to_mol = {}
        for key, smi in smiles_dict.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:  # 确保SMILES字符串有效
                smiles_to_mol[key] = mol
        return smiles_to_mol

    def seq(self, sequence):
        """根据提供的序列标识符生成指纹"""
        # 处理序列中的每个字符
        fps = []
        for char in sequence:
            mol = self.smiles_to_mol.get(char)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nbits)
                # 将RDKit的ExplicitBitVect转换为NumPy数组
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                # 如果字符在SMILES字典中未找到，添加全零向量
                fps.append(np.zeros((self.nbits,)))
        # 合并所有字符的指纹
        fp_seq = np.array(fps)
        return fp_seq