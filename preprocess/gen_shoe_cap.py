import json
import os
import glob
import nltk
import spacy
from collections import Counter
import numpy as np
import pickle
glove = spacy.load('en_vectors_web_lg').vocab

basedir = '../dataset/shoe/'
readpath = basedir+'relative_captions_shoes.json'

writecap = [basedir+'captions/cap.shoe.train.json',
            basedir+'captions/cap.shoe.val.json']
writesplit = [basedir+'image_splits/split.shoe.train.json',
              basedir+'image_splits/split.shoe.val.json']

if not os.path.exists(basedir + 'captions'):
    os.makedirs(basedir + 'captions')
if not os.path.exists(basedir + 'image_splits'):
    os.makedirs(basedir + 'image_splits')

img_txt_files = [basedir+'train_im_names.txt',
                 basedir+'eval_im_names.txt']

folder = basedir+'images'

imgfolder = os.listdir(folder)
imgfolder = [imgfolder[i] for i in range(len(imgfolder)) if 'womens' in imgfolder[i]]

### the whole path of each file
id2path = {}
for i in range(len(imgfolder)):
    path = os.path.join(folder, imgfolder[i])
    imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
    for imgname in imgfiles:
        id2path[os.path.basename(imgname)] = imgname
        
"""Process the txt files """
for ind in range(2):
    print("process", ind)
    text_file = open(img_txt_files[ind], "r")
    imgnames = text_file.readlines()
    imgnames = [imgname.strip('\n') for imgname in imgnames]

    """Process the json files """

    with open(readpath) as handle:
        dictdump = json.loads(handle.read())


    def path2id(path):
        p = path.split('/')
        return '/'.join(p[-3:]).replace('.jpg','')
    cap_data = []
    img_splits = [path2id(id2path[imn]) for imn in imgnames]
    for k in range(len(dictdump)):
        if dictdump[k]['ImageName'] in imgnames or dictdump[k]['ReferenceImageName'] in imgnames:
            target_path = id2path[dictdump[k]['ImageName']]
            source_path = id2path[dictdump[k]['ReferenceImageName']]
            target_id = path2id(target_path)
            source_id = path2id(source_path)
            text = dictdump[k]['RelativeCaption'].strip()
            cap_data.append({"target": target_id,
                             "candidate": source_id,
                             "captions": text})
            ### Following codes in VAL.
            if dictdump[k]['ImageName'] in imgnames and dictdump[k]['ReferenceImageName'] in imgnames:
                cap_data.append({"target": target_id,
                                 "candidate": source_id,
                                 "captions": text})
            if target_id not in img_splits:
                img_splits.append(target_id)
            if source_id not in img_splits:
                img_splits.append(source_id)
    print('cap', len(cap_data), 'split', len(img_splits))
    with open(writecap[ind], 'w') as f:
        json.dump(cap_data, f)
    with open(writesplit[ind], 'w') as f:
        json.dump(img_splits, f) 

# -- Glove ------------
# get words from train caps
counter = Counter()
for cap_file in [writecap[0]]:
    with open(cap_file, 'r') as f:
        cap_data = json.load(f)
    for data in cap_data:
        caption = data['captions']
        # ----- Split hyphen ------------
        caption = caption.replace('-', ' ').replace('.','') 
        toks = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(toks)
print(len(counter))
vocab_words, no_word = {}, []
for word, cnt in counter.items():
    if glove.has_vector(word):
        vocab_words[word] = glove.get_vector(word)
    elif cnt > 2:
        no_word.append(word)
        vocab_words[word] = np.random.normal(0, 0.3, (300,))
print(len(vocab_words), len(no_word))

# Save vocab
with open(basedir+'captions/glove_shoe_vecs.pkl','wb') as f:
    pickle.dump(vocab_words, f)
    
# Build glove file
for cap_file in writecap:
    # Load data
    with open(cap_file, 'r') as f:
        cap_data = json.load(f)
    # Save name
    sn = os.path.basename(cap_file).split('.')
    sn = '.'.join(sn[:2] + ["glove"] + sn[2:]).replace('.json', '.pkl')
    save_file = os.path.join(os.path.dirname(cap_file), sn)
    # Process
    w2v_data = []
    for data in cap_data:
        caption = data['captions']
        w2v = []
        caption = caption.replace('-', ' ').replace('.','') # Split hyphen
        toks = nltk.tokenize.word_tokenize(caption.lower())
        for word in toks:
            ### ------ Drop UNK words ------------------
            if word in vocab_words:
                w2v.append(vocab_words[word])
        w2v = np.stack(w2v)
        data['wv'] = w2v
    print(save_file)
    with open(save_file, 'wb') as f:
        pickle.dump(cap_data, f)
