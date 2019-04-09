import json
import urllib.request
import os

root = '/home/dongyu/桌面/faceData'

"""
    source:
        https://www.kaggle.com/dataturks/face-detection-in-images
"""
json_dir = root + '/face_detection.json'

f = open(json_dir, 'r', encoding='utf-8')

line = f.readline()

img_dir = root + '/JPEGImages'

img_id = 0


train_txt = root + '/train.txt'
test_txt = root + '/test.txt'
txt_write = open(train_txt, 'w', encoding='utf-8')

while line:
    if img_id == 300:
        txt_write.close()
        txt_write = open(test_txt, 'w', encoding='utf-8')

    file = json.loads(line)
    content = file['content']

    print('Downloading %d.jpg: %s' % (img_id,content))

    if content.endswith('png'):
        ending = 'png'
    else:
        ending = 'jpg'
    urllib.request.urlretrieve(file['content'], img_dir + '/%s.' % img_id + ending)

    txt_write.write(img_dir + '/%s.' % img_id + ending + '\n')

    now_txt = root + '/labels/' + str(img_id) + '.txt'
    g = open(now_txt, 'w', encoding='utf-8')
    for iter in file['annotation']:
        """
            xmin,xmax,ymin,ymax:[0,1]
        """
        xmin = iter['points'][0]['x']
        xmax = iter['points'][1]['x']
        ymin = iter['points'][0]['y']
        ymax = iter['points'][1]['y']
        x,y,w,h = (xmin + xmax)/2, (ymin + ymax)/2, xmax-xmin, ymax-ymin
        g.write(str(0) + " " + " ".join([str(x), str(y), str(w), str(h)]) + '\n')
    g.close()
    img_id = img_id + 1
    line = f.readline()

f.close()
txt_write.close()