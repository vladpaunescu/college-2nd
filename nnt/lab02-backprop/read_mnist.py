import cPickle
import gzip


def read_images(filename):
  # Load the dataset
  f = gzip.open(filename, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  return [train_set, valid_set, test_set]


if __name__ == "__main__":
  [train_set, valid_set, test_set] = read_images('/home/vlad/Documents/datasets/mnist.pkl.gz')
  print train_set[0].shape, train_set[1].shape
  print len(train_set[0][0])


