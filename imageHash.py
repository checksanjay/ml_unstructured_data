from PIL import Image
import imagehash
import os

script_dir = os.getcwd() + '/ml/datasets'
img2 = 'img2.jpg'
img3 = 'img3.jpg'

img2File = os.path.join(script_dir, img2)
img3File = os.path.join(script_dir, img3)

hash = imagehash.average_hash(Image.open(img2File))
print(hash)
otherhash = imagehash.average_hash(Image.open(img3File))
print(otherhash)
print(hash == otherhash)
print(hash - otherhash)