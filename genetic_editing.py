import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from cairosvg import svg2png
import time
import matplotlib.pyplot as plt
import copy
import string
import random
import re
from IPython.display import clear_output

class GeneticEditing:
    def __init__(self, img_path,svg_path, seed=0):
        self.img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.svg_path = svg_path
        self.myDNA = None
        self.seed = seed

    def generate(self, generations=100, show_progress_imgs=True):
        self.myDNA = DNA()
        self.myDNA.init(self.img_grey, self.svg_path, self.seed + time.time())
        #evolve DNA
        for g in range(generations):
            self.myDNA.evolveDNASeq(self.img_grey, self.seed + time.time() + g)
            clear_output(wait=True)
            print(". Generation ", g+1, "/", generations)
            if show_progress_imgs is True:
                #plt.imshow(sampling_mask, cmap='gray')
                print(self.myDNA.cached_error)
                plt.imshow(self.myDNA.get_cached_image(), cmap='gray')
                plt.show()
        return self.myDNA.get_cached_image()

    '''
    we'd like to "guide" the brushtrokes along the image gradient direction, if such direction has large magnitude
    in places of low magnitude, we allow for more deviation from the direction.
    this function precalculates angles and their magnitudes for later use inside DNA class
    '''

class DNA:

    def __init__(self):
        self.DNASeq = []
        #CACHE
        self.cached_image = None
        self.cached_error = None

    def init(self, target_image, svg_path, seed):
        fin = open(svg_path,'r')
        path_ = fin.read().split('d="')[1]
        path = path_.split('" stroke')[0]
        path_splited = re.split(r"([MLC])", path)
        path_splited = path_splited[1:]
        command = []
        for i in range(len(path_splited)):
            if i%2 == 0:
                command = []
                command.append(path_splited[i])
            elif i%2 == 1:
                arg = path_splited[i].split()
                for j in range(len(arg)):
                    command.append(eval(arg[j]))
                self.DNASeq.append(command)
        #calculate cache error and image
        self.cached_error, self.cached_image = self.calcTotalError(target_image)

    def get_cached_image(self):
        return self.cached_image

    def calcTotalError(self, target_image):
        return self.__calcError(self.DNASeq, target_image)

    def __calcError(self, DNASeq, target_image):
        #draw the DNA
        myImg = self.drawAll(DNASeq)

        #compare the DNA to img and calc fitness only in the ROI
        diff1 = cv2.subtract(target_image, myImg) #values are too low
        diff2 = cv2.subtract(myImg,target_image) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        totalDiff = np.sum(totalDiff)
        return (totalDiff, myImg)

    def draw(self):
        myImg = self.drawAll(self.DNASeq)
        return myImg

    def drawAll(self, DNASeq):
        svg_seq = ''
        for i in range(len(DNASeq)):
            for j in DNASeq[i]:
                svg_seq += str(j)
                svg_seq += ' '
        svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
        svg_data += '<path d="%s" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"/></g></svg>'%svg_seq
        png = svg2png(bytestring=svg_data)
        pil_img = Image.open(BytesIO(png))
        render_img = np.array(pil_img)
        render_img = render_img[:,:,-1]
        render_img = cv2.bitwise_not(render_img)
        return render_img

    def __evolveDNA(self, index, target_image, seed):
        #create a copy of the list and get its child
        DNASeqCopy = copy.deepcopy(self.DNASeq)
        child = DNASeqCopy[index]

        #mutate the child
        #select which items to mutate
        random.seed(seed + index)
        if child[0] == 'L' or child[0] == 'M':
            indexOptions = [1,2]
        else:
            indexOptions = [1,2,3,4,5,6]
        changeIndices = []
        changeCount = random.randrange(1, len(indexOptions)+1)
        for i in range(changeCount):
            random.seed(seed + index + i + changeCount)
            indexToTake = random.randrange(0, len(indexOptions))
            #move it the change list
            changeIndices.append(indexOptions.pop(indexToTake))
        #mutate selected items
        np.sort(changeIndices)
        for changeIndex in changeIndices:
            delta = random.uniform(-1,1)
            child[changeIndex] += delta
        child_error, child_img = self.__calcError(DNASeqCopy, target_image)
        if child_error < self.cached_error:
            #print('mutation!', changeIndices)
            self.DNASeq[index] = child[:]
            self.cached_image = child_img
            self.cached_error = child_error

    def evolveDNASeq(self, target_image, seed):
        for i in range(len(self.DNASeq)):
            self.__evolveDNA(i, target_image, seed)


def main():
    #load the example image and set the generator for 100 stages with 20 generations each
    gen = GeneticEditing('target_image/A11.png','source_svg/A_capital.svg', seed=time.time())
    out = gen.generate(100, 20)

if __name__ == '__main__':
    main()
