
import argparse, os, random, re, sys
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help = "path to the data dir", type = str)
args = parser.parse_args()

def parse_text(text, len_table, vocab):
  """ parse text for length and add tokens to vocab

  """
  seq = list(text.decode('utf-8'))
  length = len(seq)
  if len_table.has_key(length):
    len_table[length] += 1
  else:
    len_table[length] = 1
  for tok in seq:
    tok = tok.encode('utf-8')
    if vocab.has_key(tok):
      vocab[tok] += 1
    else:
      vocab[tok] = 1

# list of vocab
vocab = data_utils.Vocab(os.path.join(args.data_dir, "vocab"), init=True)

# files for training and testing
train_file = os.path.join(args.data_dir, "train")
dev_file = os.path.join(args.data_dir, "dev")
eval_file = os.path.join(args.data_dir, "eval")
dev_prob = 0.0001
eval_prob = 0.001

new_vocab = {"_PAD":sys.maxint, "_GO":sys.maxint, "_EOS":sys.maxint, "_UNK":sys.maxint}

text_len = {}
with open(train_file, 'w') as ftrain, open(dev_file, 'w') as fdev, open(eval_file, 'w') as feval:
  for filename in os.listdir(args.data_dir):

    # skip non-data files
    if not re.match(r'data\d*', filename):
      continue

    filename = os.path.join(args.data_dir, filename)
    if os.path.isfile(filename):
      with open(filename, 'r') as f:
        for line in f:
          text = line.strip()
          # checking
          if len(text) == 0:
            continue
          try:
            text.decode('utf-8').encode('gbk')
          except:
            continue
          else:
            # process text
            parse_text(text, text_len, new_vocab)
            # write to disk
            roll = random.random()
            if roll < dev_prob:
              fdev.write(line)
            elif roll < dev_prob+eval_prob:
              feval.write(line)
            else:
              ftrain.write(line)

print(text_len)
vocab.update(new_vocab)
vocab.dump()
