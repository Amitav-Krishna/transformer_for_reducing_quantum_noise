import os, hashlib

files = sorted([f for f in os.listdir("model_checkpoints") if f.endswith(".pth")])
hashes = [hashlib.md5(open(os.path.join("model_checkpoints", f),"rb").read()).hexdigest() for f in files]
for f, h in zip(files, hashes):
    print(f, h)

