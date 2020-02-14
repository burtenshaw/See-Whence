from emoji import UNICODE_EMOJI

def is_emoji(s):
  return s in UNICODE_EMOJI
  
def get_emojis(s):
  return [i for i in s.split(' ') if is_emoji(i)]

def translate_emojis(s):
  return [UNICODE_EMOJI[e] for e in get_emojis(s)]

def clean_emojis(t_s):
  return [' '.join(e.split('_'))[1:-1] for e in t_s]

def emoji2vec(emoji_bin_path):
    from gensim.models import KeyedVectors
    return KeyedVectors.load_word2vec_format(emoji_bin_path, binary=True)

def make_emoji_vector(emjs, voc):
  vec = np.zeros((len(emjs),len(voc)))
  for i,em in enumerate(voc):
    for j,s in enumerate(emjs):
      if em in s:
        vec[j,i]+=1
  return vec