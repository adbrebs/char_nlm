import time

from model import *
from data import *


if __name__ == '__main__':
    np.random.seed(57)

    # Config
    textfile = 'tartuffe.moliere.txt'  # has to be in the same folder
    batch_size = 100
    seq_len = 6
    emb_size = 20
    n_hidden = 500
    lr = 0.1

    # Dataset
    ds = CharDataset(textfile, batch_size, seq_len, force_reset_data=False)
    size_dict = len(ds.dictionary)

    # Model
    # m = FF(seq_len, emb_size, n_hidden, size_dict, lr)
    m = RNN(seq_len, emb_size, n_hidden, size_dict, lr)

    # Training
    last = time.clock()
    for i in range(100000000):
        batch, target = ds.next_batch()

        c = m.fun_cost(batch, target)
        if not i % 1000:
            n = time.clock()
            print 'Iteration {}, time {}, cost {}'.format(i, n - last, c)
            last = n
            m.sample(ds.dictionary, ds.inv_dictionary)
            print ""
            print "-"*70
