import numpy as np
import matplotlib.pyplot as plt
import glob
# import coco_text
import imageio
import xml.etree.ElementTree as ET
import pylab
import skimage.io as io
import scipy.misc
import scipy.io
import scipy
import os
import os.path
import random
from os import listdir
random.seed(1)

# def a(rootpath, filetype, image_prefix):
#     images = []
#     filenames = glob.glob(rootpath + "*." + filetype)
#     rootpathLength = len(rootpath) + len(image_prefix)
#     filetypeLength = len(filetype) + 1

#     for i in range(len(filenames)):
#         filenames[i] = filenames[i][rootpathLength:]
#         filenames[i] = filenames[i][:-filetypeLength]
#     filenames.sort(key=int)

#     for i in range(len(filenames)):
#         filenames[i] = rootpath + image_prefix + filenames[i] + "." + filetype
#         # images.append(plt.imread(filenames[i]))
#         images.append(scipy.ndimage.imread(filenames[i], mode='RGB'))

#     return images

# def b(rootpath, seperator, textfile_prefix, filetype):
#     textfiles = []
#     filenames = glob.glob(rootpath + "*." + filetype)
#     rootpathLength = len(rootpath) + len(textfile_prefix)
#     filetypeLength = len(filetype) + 1
#     for i in range(len(filenames)):
#         filenames[i] = filenames[i][rootpathLength:]
#         filenames[i] = filenames[i][:-filetypeLength]
#     filenames.sort(key=int)
#     for i in range(len(filenames)):
#         filenames[i] = rootpath + textfile_prefix + filenames[i] + "." + filetype
#         textfile = open(filenames[i], 'r', encoding='utf-8-sig')
#         textfile_data = textfile.readlines()
#         textfile.close()
#         edited_list = []
#         for i in range(len(textfile_data)):
#             split_data = textfile_data[i].split(seperator)
#             split_data[-1] = split_data[-1].rstrip()
#             split_data[-1] = split_data[-1].lstrip(' ')
#             split_data[-1] = split_data[-1][1:-1]
#             quad = formatBB(split_data[0], split_data[1], split_data[2], split_data[3])
#             for j in range(len(quad)):
#                 quad[j] = quad[j].rstrip()
#                 quad[j] = quad[j].lstrip(' ')
#             quad.append(split_data[-1])
#             edited_list.append(' '.join(quad))
#         textfiles.append(edited_list)
#     return textfiles

# def c(rootpath, image_prefix):    
#     images = a(rootpath, 'png', image_prefix)
#     textfile = open(rootpath + 'gt.txt', 'r')
#     textfile_data = textfile.readlines()
#     textfile.close()
#     edited_list = []
#     for i in range(len(textfile_data)):
#         split_data = textfile_data[i].split(',')
#         split_data[-1] = split_data[-1].rstrip()
#         split_data[-1] = split_data[-1].lstrip(' ')
#         split_data[-1] = split_data[-1][1:-1]
#         edited_list.append(split_data[-1])
#     return images, edited_list

# def new(rootpath, identifier):
#     textfile = open(rootpath + 'gt.txt', 'r')
#     gt_data = textfile.readlines()
#     textfile.close()
#     for i in range(len(gt_data)):
#         split_data = gt_data[i].split(',')
#         split_data[-1] = split_data[-1].rstrip()
#         split_data[-1] = split_data[-1].lstrip(' ')
#         split_data[-1] = split_data[-1][1:-1]
#         if "/" not in split_data[-1]:
#             openpath = rootpath + split_data[0]
#             img = scipy.ndimage.imread(openpath, mode='RGB')
#             savename = "./TRAINING/REAL_RECOGNITION/new/" + str(i) + "_" + split_data[-1] + identifier + ".png"
#             scipy.misc.imsave(savename, img)

# def d(rootpath, seperator, filetype, textfile_prefix):
#     textfiles = []
#     filenames = glob.glob(rootpath + "*." + filetype)
#     rootpathLength = len(rootpath) + len(textfile_prefix)
#     filetypeLength = len(filetype) + 1
#     for i in range(len(filenames)):
#         filenames[i] = filenames[i][rootpathLength:]
#         filenames[i] = filenames[i][:-filetypeLength]
#     filenames.sort(key=int)
#     for i in range(len(filenames)):
#         filenames[i] = rootpath + textfile_prefix + filenames[i] + "." + filetype
#         textfile = open(filenames[i], 'r', encoding='utf-8-sig')
#         textfile_data = textfile.readlines()
#         textfile.close()
#         edited_list = []
#         for i in range(len(textfile_data)):
#             split_data = textfile_data[i].split(seperator)
#             split_data[0] = split_data[0].replace('\ufeff', '')
#             split_data[-1] = split_data[-1].rstrip()
#             split_data[-1] = split_data[-1].lstrip(' ')
#             edited_list.append(' '.join(split_data))
#         textfiles.append(edited_list)
#     return textfiles


def xmlFormatBB(bb):
	p1 = [float(bb[0]), float(bb[1])]
	p2 = [float(bb[2]), float(bb[3])]
	p3 = [float(bb[4]), float(bb[5])]
	p4 = [float(bb[6]), float(bb[7])]

	'''
	the box coordinates seem to be in different positions, so i need to write some logic that will format them all
	the same regardless of there current orientation
	'''

	# print(bb)

	finalBB = []

	# get top left point -----
	candidatePoints = [p1, p2, p3, p4]
	candidatePointsX = [p1[0], p2[0], p3[0], p4[0]]
	# candidatePointsY = [p1[1], p2[1], p3[1], p4[1]]

	# print(candidatePoints)
	# print(candidatePointsX)

	target = []
	minX = min(candidatePointsX)
	for point in candidatePoints:
		if point[0] == minX:
			target.append(point)

	# print(target)
	if len(target) > 1:
		ylist = []
		for elem in target:
			ylist.append(elem[1])
		minY = min(ylist)
		# print(minY)
		target = sorted(target,key=lambda l:l[1])
		for item in target:
			# print(item, minY)
			if item[1] == minY:
				# print("selected ", item)
				if item == p1:
					finalBB.append(p1)
				elif item == p2:
					finalBB.append(p2)
				elif item == p3:
					finalBB.append(p3)
				else:
					finalBB.append(p4)

				for i in candidatePoints:
					if i == item:
						candidatePoints.remove(i)
						break
				for i in candidatePointsX:
					if i == item[0]:
						candidatePointsX.remove(i)
						break  
			break   
	else:
		if target[0] == p1:
			finalBB.append(p1)
		elif target[0] == p2:
			finalBB.append(p2)
		elif target[0] == p3:
			finalBB.append(p3)
		else:
			finalBB.append(p4)

		for i in candidatePoints:
			if i == target[0]:
				candidatePoints.remove(i)
				break
		for i in candidatePointsX:
			if i == target[0][0]:
				candidatePointsX.remove(i)
				break     

	# print(finalBB)
	# print("\n")
	# ----------------------

	target = []
	minX = min(candidatePointsX)
	for point in candidatePoints:
		if point[0] == minX:
			target.append(point)
	if len(target) > 1:
		ylist = []
		for elem in target:
			ylist.append(elem[1])
		minY = min(ylist)
		target = sorted(target,key=lambda l:l[1])
		for item in target:
			if item[1] == minY:
				if item == p1:
					finalBB.append(p1)
				elif item == p2:
					finalBB.append(p2)
				elif item == p3:
					finalBB.append(p3)
				else:
					finalBB.append(p4)

				for i in candidatePoints:
					if i == item:
						candidatePoints.remove(i)
						break
				for i in candidatePointsX:
					if i == item[0]:
						candidatePointsX.remove(i)
						break    
			break     
	else:
		if target[0] == p1:
			finalBB.append(p1)
		elif target[0] == p2:
			finalBB.append(p2)
		elif target[0] == p3:
			finalBB.append(p3)
		else:
			finalBB.append(p4)

		for i in candidatePoints:
			if i == target[0]:
				candidatePoints.remove(i)
				break
		for i in candidatePointsX:
			if i == target[0][0]:
				candidatePointsX.remove(i)
				break     
# ------------------------------------------

	candidatePointsY = []
	for i in candidatePoints:
		candidatePointsY.append(i[1])

	targetX = max(finalBB[0][0], finalBB[1][0])
	maxX = max(candidatePointsX)
	if candidatePointsX[0] > targetX and candidatePointsX[1] > targetX:
		for point in candidatePoints:
			if point[0] == maxX:
				target.append(point)
		finalBB.append(target[0])
		candidatePoints.remove(target[0])

	if finalBB[1][1] < finalBB[0][1]:
		minY = min(candidatePointsY)
	else:
		minY = max(candidatePointsY)

	target = []
	
	# print(candidatePoints)
	# print(candidatePointsY)
	# print(minY)
	for point in candidatePoints:
		if point[1] == minY:
			target.append(point)
	finalBB.append(target[0])
	candidatePoints.remove(target[0])
# -----------------------------------------

	finalBB.append(candidatePoints[0])
	finalBB = [item for sublist in finalBB for item in sublist]
	for i in range(len(finalBB)):
		finalBB[i] = str(finalBB[i])

	return finalBB

def formatBB(a, b, c, d):
	x1 = a
	y1 = b
	x2 = c
	y2 = b
	x3 = a
	y3 = d
	x4 = c
	y4 = d
	return [x1, y1, x3, y3, x4, y4, x2, y2]

def loadVideo(rootpath, filetype):
	vids = []
	for filename in glob.glob(rootpath + "*." + filetype):
		vids.append(imageio.get_reader(filename,  'ffmpeg'))
	return vids

def parseXMLgt(rootpath):
	vids = []
	for filename in glob.glob(rootpath + "*.xml"):
		tree = ET.parse(filename)
		frames = tree.getroot()
		res = []
		for frame in frames:
			boundingBoxesInFrame = []
			for obj in frame:
				boundingBox = []
				for point in obj:
					boundingBox.append(point.get('x'))
					boundingBox.append(point.get('y'))
				boundingBox = xmlFormatBB(boundingBox)
				boundingBox.append(obj.get('Transcription'))
				boundingBoxesInFrame.append(' '.join(boundingBox))
			res.append(boundingBoxesInFrame)
		vids.append(res)
	return vids

# def loadCoco(startIndex, batchSize):
#     ct = coco_text.COCO_Text('COCO_Text.json')
#     dataDir='.'
#     dataType='icdar_2017_coco_text/training'
#     imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible'), ('language', 'english')])
#     labels = []
#     images = []
#     print(len(imgIds))
#     if startIndex + batchSize < len(imgIds):
#         endIndex = startIndex + batchSize
#     else:
#         endIndex = len(imgIds)

#     for i in range(startIndex, endIndex):
#         img = ct.loadImgs(imgIds[i])[0]
#         I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
#         images.append(I)
#         annIds = ct.getAnnIds(imgIds=img['id'])
#         anns = ct.loadAnns(annIds)
#         image_labels = []
#         for anno in anns:
#             if 'utf8_string' in anno:
#                 if anno['language'] == 'english':
#                     text = str(anno['utf8_string'])
#                     polygon = list(anno['polygon'])
#                     for i in range(len(polygon)):
#                         polygon[i] = str(polygon[i])
#                     polygon.append(text)
#                     gt = ' '.join(polygon)
#                     image_labels.append(gt)
#             else:
#                 if anno['language'] == 'english':
#                     text = '#####'
#                     polygon = list(anno['polygon'])
#                     for i in range(len(polygon)):
#                         polygon[i] = str(polygon[i])
#                     polygon.append(text)
#                     gt = ' '.join(polygon)
#                     image_labels.append(gt)
#         labels.append(image_labels)
#     return images, labels

# def save_data(images, labels, savepath):
#     for i in range(len(images)):
#         image_filename = savepath + "images/coco_images_4/coco_" + str(i+12000) + ".png"
#         gt_filename = savepath + "gt/" + "coco_" + str(i+12000) + ".txt"

#         scipy.misc.imsave(image_filename, images[i])
#         # scipy.misc.toimage(images[i]).save(image_filename)

#         # with open(gt_filename, 'w') as file:
#         #     file.write(labels[i])
#         #     file.write('\n')

#         with open(gt_filename, 'w') as file:
#             if len(labels[i]) > 1:
#                 for label in labels[i]:
#                     file.write(label)
#                     file.write('\n')
#             else:
#                 if labels[i]:
#                     file.write(labels[i][0])
#                     file.write('\n')

# # Call this immediately after loading raw icdar recognition data
# def process_rec_data(images, labels):
#     edited_rec_training_images = []
#     edited_rec_training_gt = []
#     for i in range(len(images)):
#         if images[i].shape[2] == 4:
#             edited_rec_training_images.append((images[i][:, :, :3] * 255).astype(int))
#         else:
#             edited_rec_training_images.append((images[i] * 255).astype(int))
#         if len(labels) > 0:
#             edited_rec_training_gt.append([labels[i]])
#     return edited_rec_training_images, edited_rec_training_gt


# def getSynthTextLabels():
#     mat = scipy.io.loadmat('./SynthText/SynthText/gt.mat')
#     wordBBs = []
#     text = []
#     imnames = []
#     charBBs = []
#     for key, value in mat.items():
#         if key == 'wordBB':
#             wordBBs.append(value)
#         if key == 'txt':
#             text.append(value)
#         if key == 'imnames':
#             imnames.append(value)
#         if key == 'charBBs':
#             charBBs.append(value)
#     textfiles = imnames[0][0]
#     textlabels = text[0][0]
#     bbs = wordBBs[0][0]
#     prefix = './SynthText/SynthText/'
#     all_labels = []
#     all_paths = []
#     for i in range(len(textfiles)):
#         # path = prefix + textfiles[i][0]
#         split = textfiles[i][0].split('/')
#         name = split[-1]
#         name = name[:-4]
#         all_paths.append(name)
#         bbs_for_this_image = bbs[i]
#         Xs = bbs_for_this_image[0]
#         Ys = bbs_for_this_image[1]
#         labels = []
#         if not isinstance(Xs[0], np.ndarray):
#             points = []
#             for j in range(4):
#                 points.append(str(Xs[j]))
#                 points.append(str(Ys[j]))
#             points = ' '.join(points)
#             labels.append(points)
# #             img = plt.imread(path)
#             words_in_pic = []
#             for line in textlabels[i]:
#                 line = line.strip()
#                 for group in line.split(' '):
#                     group = group.strip()
#                     group = group.split('\n')
#                     for word in group:
#                         words_in_pic.append(word)
#             for x in range(len(labels)):
#                 labels[x] = labels[x] + ' ' + words_in_pic[x]
#             all_labels.append(labels) 
#         else:
#             for k in range(len(Xs[0])):
#                 points = []
#                 for j in range(4):
#                     points.append(str(Xs[j][k]))
#                     points.append(str(Ys[j][k]))
#                 points = ' '.join(points)
#                 labels.append(points)
#             # img = plt.imread(path)

#             words_in_pic = []
#             for line in textlabels[i]:
#                 line = line.strip()
#                 for group in line.split(' '):
#                     group = group.strip()
#                     group = group.split('\n')
#                     for word in group:
#                         words_in_pic.append(word)
#             for x in range(len(labels)):
#                 labels[x] = labels[x] + ' ' + words_in_pic[x]
#             all_labels.append(labels)
#     return all_labels, all_paths

# def getSVT(rootpath):
#     for filename in glob.glob(rootpath + "train.xml"):
#         tree = ET.parse(filename)
#         tagset = tree.getroot()
#         images = []
#         labels = []
#         for image in tagset:
#             for characteristic in image:
#                 if characteristic.tag == 'imageName':
#                     imagePath = rootpath + str(characteristic.text)
#                 if characteristic.tag == 'taggedRectangles':
#                     rects_per_image = []
#                     for rect in characteristic:
#                         height = int(rect.get('height'))
#                         width = int(rect.get('width'))
#                         x = int(rect.get('x'))
#                         y = int(rect.get('y'))
#                         tag = str(rect[0].text)
#                         quad = svtFormat(height, width, x, y, tag)
#                         rects_per_image.append(quad)
#             images.append(scipy.ndimage.imread(imagePath, mode='RGB'))
#             labels.append(rects_per_image)
#     return images, labels

# def svtFormat(height, width, x, y, tag):
#     return ' '.join([str(x), str(y), str(x+width), str(y), str(x+width), str(y+height), str(x), str(y+height), tag])


# def reformat_rec_data():
#     image_names = glob.glob("./TRAINING/REAL_RECOGNITION/images/*.png")
#     gt_names = glob.glob("./TRAINING/REAL_RECOGNITION/gt/*.txt")
#     image_names.sort()
#     gt_names.sort()
#     for i in range(len(image_names)):
#         img = scipy.ndimage.imread(image_names[i], mode='RGB')
#         textfile = open(gt_names[i], 'r')
#         gtdata = textfile.readlines()
#         textfile.close()
#         gt = gtdata[0].rstrip()
#         print(gt, gt_names[i])

#         # path = "./TRAINING/REAL_RECOGNITION/new_images/"        
#         # savepath = path + gt + ".png"
#         # print(savepath)
#         # scipy.misc.imsave(savepath, img)





# ===================================== Load Training Data Data =====================================

def load_training_real_detection_datasets(startIndex, batchSize):
	image_names = glob.glob("./TRAINING/REAL_DETECTION/images/*.png")
	gt_names = glob.glob("./TRAINING/REAL_DETECTION/gt/*.txt")
	image_names.sort()
	gt_names.sort()

	if startIndex + batchSize < len(image_names):
		endIndex = startIndex + batchSize
	else:
		endIndex = len(image_names)

	images = []
	gt = []
	for i in range(startIndex, endIndex):
		images.append(scipy.ndimage.imread(image_names[i], mode='RGB'))
		textfile = open(gt_names[i], 'r')
		textfile_data = textfile.readlines()
		textfile.close()
		new_textfile_data = []
		for l in range(len(textfile_data)):
			if len(textfile_data[l].rstrip()) > 0:
				new_textfile_data.append(textfile_data[l].rstrip())
		gt.append(new_textfile_data)
	return images, gt
	
def load_training_synthetic_detection_datasets(startIndex, batchSize):
	gt_names = glob.glob("./TRAINING/SYNTHETIC_DETECTION/gt/*.txt")
	gt_names.sort()
	image_names = []
	for dirpath, dirnames, filenames in os.walk("./TRAINING/SYNTHETIC_DETECTION/images/"):
		for filename in [f for f in filenames if f.endswith(".jpg")]:
			image_names.append((dirpath, filename))

	image_names.sort(key=lambda x: x[1])

	if startIndex + batchSize < len(image_names):
		endIndex = startIndex + batchSize
	else:
		endIndex = len(image_names)

	images = []
	gt = []
	for i in range(startIndex, endIndex):
		image_path = image_names[i][0] + "/" + image_names[i][1]
		images.append(scipy.ndimage.imread(image_path, mode='RGB'))
		textfile = open(gt_names[i], 'r')
		textfile_data = textfile.readlines()
		textfile.close()
		for l in range(len(textfile_data)):
			textfile_data[l] = textfile_data[l].rstrip()
		gt.append(textfile_data)
	return images, gt

# THIS SHOULD RETURN SHUFFLED DATA AS THIS WILL NOT BE AUGMENTED!
def load_training_real_recognition_datasets(startIndex, batchSize):
	# random.seed(1)
	image_names = glob.glob("./TRAINING/REAL_RECOGNITION/images/*.png")
	random.shuffle(image_names)
	images = []
	gt = []

	if startIndex + batchSize < len(image_names):
		endIndex = startIndex + batchSize
	else:
		endIndex = len(image_names)

	for i in range(startIndex, endIndex):
		images.append(scipy.ndimage.imread(image_names[i], mode='RGB'))
		title = image_names[i].split("_")
		gt.append(title[-2])

	return images, gt

def load_training_synthetic_recognition_datasets(startIndex, batchSize, paths):
	images = []
	gt = []

	if startIndex + batchSize < len(paths):
		endIndex = startIndex + batchSize
	else:
		endIndex = len(paths)

	for i in range(startIndex, endIndex):
		images.append(scipy.ndimage.imread(paths[i], mode='RGB'))
		title = paths[i].split("_")
		gt.append(title[-2])

	return images, gt

def get_array_of_image_paths():
	# random.seed(1)

	image_names = []
	for dirpath, dirnames, filenames in os.walk("./TRAINING/SYNTHETIC_RECOGNITION/images/"):
		for filename in [f for f in filenames if f.endswith(".jpg")]:
			image_names.append(os.path.join(dirpath, filename))

	random.shuffle(image_names)

	return image_names

# ===================================== Load Training Data Data =====================================

def load_testing_real_detection_datasets():
	image_names_2013 = glob.glob("./TESTING/REAL_DETECTION/icdar2013/images/*.png")
	image_names_2015 = glob.glob("./TESTING/REAL_DETECTION/icdar2015/images/*.png")

	images_2013 = []; images_2015 = []

	rootpathLength = len("./TESTING/REAL_DETECTION/icdar2013/images/") + 4
	filetypeLength = 4

	for i in range(len(image_names_2013)):
		image_names_2013[i] = image_names_2013[i][rootpathLength:]
		image_names_2013[i] = image_names_2013[i][:-filetypeLength]
	image_names_2013.sort(key=int)

	for i in range(len(image_names_2013)):
		image_names_2013[i] = "./TESTING/REAL_DETECTION/icdar2013/images/img_" + image_names_2013[i] + ".png"
		images_2013.append(scipy.ndimage.imread(image_names_2013[i], mode='RGB'))


	rootpathLength = len("./TESTING/REAL_DETECTION/icdar2015/images/") + 4
	filetypeLength = 4

	for i in range(len(image_names_2015)):
		image_names_2015[i] = image_names_2015[i][rootpathLength:]
		image_names_2015[i] = image_names_2015[i][:-filetypeLength]
	image_names_2015.sort(key=int)

	for i in range(len(image_names_2015)):
		image_names_2015[i] = "./TESTING/REAL_DETECTION/icdar2015/images/img_" + image_names_2015[i] + ".png"
		images_2015.append(scipy.ndimage.imread(image_names_2015[i], mode='RGB'))

	return images_2013, image_names_2013, images_2015, image_names_2015

def load_testing_real_recognition_datasets():
	image_names_2013 = glob.glob("./data/icdar_2013_focused_scene_text/recognition/testing/*.png")
	image_names_2015 = glob.glob("./data/icdar_2015_incidental_scene_text/recognition/testing/*.png")

	images_2013 = []; images_2015 = []

	rootpathLength = len("./data/icdar_2013_focused_scene_text/recognition/testing/") + 5
	filetypeLength = 4

	for i in range(len(image_names_2013)):
		image_names_2013[i] = image_names_2013[i][rootpathLength:]
		image_names_2013[i] = image_names_2013[i][:-filetypeLength]
	image_names_2013.sort(key=int)

	for i in range(len(image_names_2013)):
		image_names_2013[i] = "./data/icdar_2013_focused_scene_text/recognition/testing/word_" + image_names_2013[i] + ".png"
		images_2013.append(scipy.ndimage.imread(image_names_2013[i], mode='RGB'))


	rootpathLength = len("./data/icdar_2015_incidental_scene_text/recognition/testing/") + 5
	filetypeLength = 4

	for i in range(len(image_names_2015)):
		image_names_2015[i] = image_names_2015[i][rootpathLength:]
		image_names_2015[i] = image_names_2015[i][:-filetypeLength]
	image_names_2015.sort(key=int)

	for i in range(len(image_names_2015)):
		image_names_2015[i] = "./data/icdar_2015_incidental_scene_text/recognition/testing/word_" + image_names_2015[i] + ".png"
		images_2015.append(scipy.ndimage.imread(image_names_2015[i], mode='RGB'))

	return images_2013, image_names_2013, images_2015, image_names_2015

def read_optical_flow_datasets():
	rootpath = "./FlowDatasets/FlyingChairs/data/"
	files = os.listdir(rootpath)
	files.sort()
	counter = 0
	collection = []
	all_collections = []
	for i in range(len(files)):
		# file = files[i].lstrip("0")
		if counter == 0:
			collection.append(files[i])
			counter+=1
		elif counter == 1:
			collection.append(files[i])
			counter+=1
		else:
			collection.append(files[i])
			all_collections.append(collection)
			collection = []
			counter = 0

	# for collection in all_collections:
	# 	print(collection)

	flow = all_collections[0][0]
	image1 = all_collections[0][1]
	image2 = all_collections[0][2]

	img = scipy.ndimage.imread(rootpath + image1, mode='RGB')
	fig, ax = plt.subplots(1)
	ax.imshow(img)
	img = scipy.ndimage.imread(rootpath + image2, mode='RGB')
	fig, ax = plt.subplots(1)
	ax.imshow(img)











































