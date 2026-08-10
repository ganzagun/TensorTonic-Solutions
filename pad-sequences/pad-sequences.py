import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    N  = len(seqs)
    if max_len is None:
        max_len = 0
        for i in range(N):
            max_len = max(max_len, len(seqs[i]))

    pad_seq = np.full((N, max_len), pad_value)

    for i in range(N):
        lenSeq = min(max_len, len(seqs[i]))
        
        pad_seq[i, :lenSeq] = seqs[i][:lenSeq]

    return pad_seq
    

    