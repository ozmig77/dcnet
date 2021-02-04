import subprocess
from tqdm import tqdm

txt_paths = ['../dataset/fashioniq/asin2url.dress.txt',
             '../dataset/fashioniq/asin2url.shirt.txt',
             '../dataset/fashioniq/asin2url.toptee.txt']

for t_p in txt_paths:
    with open(t_p) as f:
        lines = f.readlines()
    for line in lines:
        iid, url = line.strip().split('\t')
        iid, url = iid.strip(), url.strip()
        name = '../dataset/fashioniq/images/%s.jpg'%iid
        subprocess.call(["wget", "-O", name, url], stdout=subprocess.PIPE)