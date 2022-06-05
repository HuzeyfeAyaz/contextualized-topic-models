import torch
from torch.utils.data import Dataset
import scipy.sparse
import numpy as np

class CTMDataset(Dataset):

    """Class to load BoW and the contextualized embeddings."""

    def __init__(self, X_contextual, X_bow, idx2token, retained_indices, labels=None):

        if labels is not None:
            if labels.shape[0] != X_bow.shape[0]:
                raise Exception(f"There is something wrong in the length of the labels (size: {labels.shape[0]}) "
                                f"and the bow (len: {X_bow.shape[0]}). These two numbers should match.")

        self.X_contextual = X_contextual
        self.X_bow = X_bow
        self.idx2token = idx2token
        self.retained_indices = retained_indices
        self.labels = labels

    def __len__(self):
        """Return length of dataset."""
        return self.X_bow.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        idx = self.retained_indices[i]
        if type(self.X_bow[i]) == scipy.sparse.csr.csr_matrix:
            X_bow = torch.FloatTensor(self.X_bow[i].todense())
            X_contextual = torch.FloatTensor(self.X_contextual[idx].astype(np.float16))
        else:
            X_bow = torch.FloatTensor(self.X_bow[i])
            X_contextual = torch.FloatTensor(self.X_contextual[idx].astype(np.float16))

        return_dict = {'X_bow': X_bow, 'X_contextual': X_contextual}

        if self.labels is not None:
            labels = self.labels[i]
            if type(labels) == scipy.sparse.csr.csr_matrix:
                return_dict["labels"] = torch.FloatTensor(labels.todense())
            else:
                return_dict["labels"] = torch.FloatTensor(labels)

        return return_dict