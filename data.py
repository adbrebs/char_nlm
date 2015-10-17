import os
import cPickle
import numpy as np


def create_dictionary(textfile, force=False):
    """
    Create dictionary {character: index} from a text file and save it
    in the same folder.

    Parameters
    ----------
    textfile: str
        the name of a textfile that is located in the same folder.
        ex: raccoon.txt
    force: boolean
        if True, forces the function to re-create the dictionary even if
        the dictionary file already exists
    """
    outfile = 'dict_' + textfile

    if force or not os.path.isfile(outfile):
        data = open(textfile, 'r').read()
        dict = {ch: i for i, ch in enumerate(list(set(data)))}
        cPickle.dump(dict, open(outfile, 'w'))
    else:
        dict = cPickle.load(open(outfile, 'r'))

    return dict


def int2char(ints, dictionary):
    """
    Converts a list of integer into a string of letters.
    """
    inv_dict = {i: ch for ch,i in dictionary.iteritems()}
    ints = list(ints)
    out = ""
    for i in ints:
        out += inv_dict[i]
    return out


def create_data(textfile, dictionary, force=False):
    """
    Converts string dataset into an array of integer indices.
    """
    outfile = 'data_' + textfile

    if force or not os.path.isfile(outfile):
        data = open(textfile, 'r').read()
        data = np.array([dictionary[ch] for ch in data], dtype='int16')
        cPickle.dump(data, open(outfile, 'w'))
    else:
        data = cPickle.load(open(outfile, 'r'))

    return data


class CharDataset:
    """
    Dataset which manages batch pointers in the data. These pointers read text
    at different locations of the file. Each pointer represents a datastream
    for each sample in a minibatch. Thus there are as many pointers as
    minibatch.
    """
    def __init__(self, textfile, batch_size, seq_len, force_reset_data=False):

        self.dictionary = create_dictionary(textfile,
                                            force=force_reset_data)
        self.inv_dictionary = {i: ch for ch, i in self.dictionary.iteritems()}
        self.data = create_data(textfile, self.dictionary,
                                force=force_reset_data)
        self.data_size = len(self.data)

        self.batch_size = batch_size
        self.seq_len = seq_len

        # Pointers are evenly initialized in the dataset.
        self.pointers = np.arange(0, batch_size, dtype='int32') * (
            self.data_size - seq_len - 1) / batch_size

    def next_batch(self):

        batch = np.zeros((self.batch_size, self.seq_len), dtype='int16')
        target = np.zeros((self.batch_size,), dtype='int16')

        ps = self.pointers
        for i in xrange(self.pointers.shape[0]):
            p = ps[i]
            batch[i] = self.data[p:p+self.seq_len]
            target[i] = self.data[p+self.seq_len]

        self.pointers = (self.pointers + 1) % (self.data_size-self.seq_len-1)

        return batch, target