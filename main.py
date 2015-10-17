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
    print_frequency = 10000

    # Dataset
    ds = CharDataset(textfile, batch_size, seq_len, force_reset_data=False)
    size_dict = len(ds.dictionary)

    # Model
    m = FF(seq_len, emb_size, n_hidden, size_dict, lr)
    # m = RNN(seq_len, emb_size, n_hidden, size_dict, batch_size, lr)

    # Training
    last = time.clock()
    cost = .0
    for i in range(100000000):
        batch, target = ds.next_batch()

        cost += m.train(batch, target, i)
        if not i % print_frequency:
            n = time.clock()
            print 'Iteration {}, time {}, cost {}'.format(
                i, n - last, cost/print_frequency)
            cost = 0
            last = n
            m.sample(ds.dictionary, ds.inv_dictionary, seq_print_len=1000)
            print ""
            print "-"*70
