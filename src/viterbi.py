import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


train = pd.read_csv('../data/cleaned/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
test = pd.read_csv('../data/cleaned/test.csv', dtype={'time': np.float32, 'signal': np.float32})


class ViterbiClassifier:
    # https://www.kaggle.com/miklgr500/viterbi-algorithm-without-segmentation-on-groups
    def __init__(self, num_bins=1000):
        self._n_bins = num_bins
        self._p_trans = None
        self._p_signal = None
        self._signal_bins = None
        self._p_in = None
    
    def fit(self, x, y):
        self._p_trans = self.markov_p_trans(y)
        self._p_signal, self._signal_bins = self.markov_p_signal(true_state, x, self._n_bins)
        self._p_in = np.ones(len(self._p_trans)) / len(self._p_trans)
        return self
        
    def predict(self, x):
        x_dig = self.digitize_signal(x, self._signal_bins)
        return self.viterbi(self._p_trans, self._p_signal, self._p_in, x_dig)
    
    @classmethod
    def digitize_signal(cls, signal, signal_bins):
        signal_dig = np.digitize(signal, bins=signal_bins) - 1 # these -1 and -2 are necessary because of the way...
        signal_dig = np.minimum(signal_dig, len(signal_bins) - 2) # ... numpy.digitize works
        return signal_dig
    
    @classmethod
    def markov_p_signal(cls, state, signal, num_bins = 1000):
        states_range = np.arange(state.min(), state.max() + 1)
        signal_bins = np.linspace(signal.min(), signal.max(), num_bins + 1)
        p_signal = np.array([ np.histogram(signal[state == s], bins=signal_bins)[0] for s in states_range ])
        p_signal = np.array([ p / np.sum(p) if np.sum(p) != 0 else p for p in p_signal ]) # normalize to 1
        return p_signal, signal_bins
    
    @classmethod
    def markov_p_trans(cls, states):
        max_state = np.max(states)
        states_next = np.roll(states, -1)
        matrix = []
        for i in range(max_state + 1):
            current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]
            if np.sum(current_row) == 0: # if a state doesn't appear in states...
                current_row = np.ones(max_state + 1) / (max_state + 1) # ...use uniform probability
            else:
                current_row = current_row / np.sum(current_row) # normalize to 1
            matrix.append(current_row)
        return np.array(matrix)
    
    @classmethod
    def viterbi(cls, p_trans, p_signal, p_in, signal):
        offset = 10**(-20) # added to values to avoid problems with log2(0)

        p_trans_tlog  = np.transpose(np.log2(p_trans  + offset)) # p_trans, logarithm + transposed
        p_signal_tlog = np.transpose(np.log2(p_signal + offset)) # p_signal, logarithm + transposed
        p_in_log      =              np.log2(p_in     + offset)  # p_in, logarithm

        p_state_log = [ p_in_log + p_signal_tlog[signal[0]] ] # initial state probabilities for signal element 0 

        for s in signal[1:]:
            p_state_log.append(np.max(p_state_log[-1] + p_trans_tlog, axis=1) + p_signal_tlog[s]) # the Viterbi algorithm

        states = np.argmax(p_state_log, axis=1) # finding the most probable states
    
        return states


true_state = train.open_channels.values
signal = train.signal.values

viterbi = ViterbiClassifier().fit(signal, true_state)
train_prediction = viterbi.predict(signal)

f1 = f1_score(y_pred=train_prediction, y_true=true_state, average='macro')
print( f"F1 macro = {f1:0.5f}" )


df_subm = pd.read_csv("../data/sample_submission.csv")
df_subm['open_channels'] = viterbi.predict(test.signal.values)
df_subm.to_csv("../data/viterbi_res/viterbi_test.csv", float_format='%.4f', index=False)

viterbi_train = train[['time','signal']].copy()
viterbi_train['viterbi_preds'] = train_prediction
viterbi_train.to_csv(f"../data/viterbi_res/viterbi_train_{f1:0.5f}.csv", index=False)