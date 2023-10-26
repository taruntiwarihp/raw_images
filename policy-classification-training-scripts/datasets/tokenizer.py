import numpy as np
import torch

class GloveTokenizer(object):

    def __init__(self, vector_file='dataset/glove/glove.6B.300d.txt'):
        self.vector_file = vector_file
        self.pad_token, self.unk_token = '<pad>','<unk>'
        self.vocab, self.embeddings = self.create_vocab()

        self.word2idx = {term:idx for idx,term in enumerate(self.vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

    def create_vocab(self):

        vocab,embeddings = [],[]
        with open(self.vector_file,'rt') as f:
            full_content = f.read().strip().split('\n')
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)

        vocab_npa = np.array(vocab)
        embs_npa = np.array(embeddings)

        vocab_npa = np.insert(vocab_npa, 0, '<pad>')
        vocab_npa = np.insert(vocab_npa, 1, '<unk>')

        pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
        unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

        #insert embeddings for pad and unk tokens at top of embs_npa.
        embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

        return vocab_npa, embs_npa

    def encode(self, text, max_len):
        words = text.strip().split()[:max_len]
        deficit = max_len - len(words)
        words.extend([self.pad_token] * deficit)

        emb_words = []
        for w in words:
            if w not in self.word2idx:
                emb_words.append(self.word2idx[self.unk_token])

            else:
                emb_words.append(self.word2idx[w])

        return torch.Tensor(emb_words).long(), max(1, max_len - deficit)


