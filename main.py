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

class SVG:
    def __init__(self,svg_seq):
        self.svg_seq = svg_seq
        self.loss = None

def util_sample_from_img(img):
    #possible positions to sample
    pos = np.indices(dimensions=img.shape)
    pos = pos.reshape(2, pos.shape[1]*pos.shape[2])
    img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
    pos = pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]
    posY = pos[0][0]
    posX = pos[1][0]
    return posX, posY


class Editer:
    def __init__(self,img_path,svg_path, pop_size, seed = 0):
        self.img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.svg_path = svg_path
        self.cached_image = None
        self.cached_error = None
        self.sample_mask = None
        self.population = [[],[]]
        self.pp = [None for i in range(pop_size)]
        self.pop_size = pop_size
        self.cur = 0
        self.height,self.width = self.img_grey.shape
        self.seed = seed


    def InitPopulation(self):
        fin = open(self.svg_path,'r')
        path_ = fin.read().split('d="')[1]
        path = path_.split('" stroke')[0]
        path_splited = re.split(r"([MLC])", path)
        path_splited = path_splited[1:]
        svg_seq = []
        command = []
        for i in range(len(path_splited)):
            if i%2 == 0:
                command = []
                command.append(path_splited[i])
            elif i%2 == 1:
                arg = path_splited[i].split()
                for j in range(len(arg)):
                    command.append(eval(arg[j]))
                svg_seq.append(command)
        for i in range(self.pop_size):
            svg = SVG(copy.deepcopy(svg_seq))
            self.population[self.cur].append(svg)
        self.population[1 - self.cur] = [None for i in range(self.pop_size)]

    def Draw(self, svg):
        svg_seq = svg.svg_seq
        svg_str = ''
        for i in range(len(svg_seq)):
            for j in svg_seq[i]:
                svg_str += str(j)
                svg_str += ' '
        svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
        svg_data += '<path d="%s" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"/></g></svg>'%svg_str
        png = svg2png(bytestring=svg_data)
        pil_img = Image.open(BytesIO(png))
        render_img = np.array(pil_img)
        render_img = render_img[:,:,-1]
        render_img = cv2.bitwise_not(render_img)
        return render_img

    def DrawSeq(self, svg_seq):
        svg_str = ''
        for i in range(len(svg_seq)):
            for j in svg_seq[i]:
                svg_str += str(j)
                svg_str += ' '
        svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
        svg_data += '<path d="%s" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"/></g></svg>'%svg_str
        png = svg2png(bytestring=svg_data)
        pil_img = Image.open(BytesIO(png))
        render_img = np.array(pil_img)
        render_img = render_img[:,:,-1]
        render_img = cv2.bitwise_not(render_img)
        return render_img

    def Evaluate(self, render_img):
        diff1 = cv2.subtract(self.img_grey, render_img) #values are too low
        diff2 = cv2.subtract(render_img,self.img_grey) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        totalDiff = np.sum(totalDiff)
        return totalDiff / (self.height * self.width)

    def EvaluatePopulation(self):
        loss_best = 1e12
        loss_worst = 0
        for i in range(self.pop_size):
            svg = self.population[self.cur][i]
            render_img = self.Draw(svg)
            svg.loss = self.Evaluate(render_img)
            if svg.loss < loss_best:
                loss_best = svg.loss
                idx_best = i
            if svg.loss > loss_worst:
                loss_worst = svg.loss
                idx_worst = i
        return idx_best, idx_worst

    def ComputeCrossProb(self, xi, loss_worst):
        sum = 0
        for i in range(self.pop_size):
            self.pp[i] = loss_worst - self.population[self.cur][i].loss + xi
            sum += self.pp[i]
        self.pp[0] /= sum
        for i in range(1, self.pop_size):
            self.pp[i] = self.pp[i-1] + self.pp[i] /sum

    def Clamp(self, min_number, max_number, parameter):
        if min_number < parameter and max_number > parameter:
            return parameter
        elif parameter <= min_number:
            return min_number
        elif parameter >= max_number:
            return max_number

    def MutatePos(self, svg, seed):
        svg_seq = svg.svg_seq
        cache_image = self.Draw(svg)
        cache_error = self.Evaluate(cache_image)
        for idx in range(len(svg_seq)):
            tmp_seq = copy.deepcopy(svg_seq)
            command = tmp_seq[idx]
            #mutate the child
            #select which items to mutate
            random.seed(seed + idx)
            if command[0] == 'L' or command[0] == 'M':
                indexOptions = [1,2]
            else:
                indexOptions = [1,2,3,4,5,6]
            changeIndices = []
            changeCount = random.randrange(1, len(indexOptions)+1)
            for i in range(changeCount):
                random.seed(seed + idx + i + changeCount)
                indexToTake = random.randrange(0, len(indexOptions))
                #move it the change list
                changeIndices.append(indexOptions.pop(indexToTake))
            #mutate selected items
            np.sort(changeIndices)
            for changeIndex in changeIndices:
                delta = random.uniform(-2,2)
                command[changeIndex] += delta
                command[changeIndex] = self.Clamp(0, 256-1, command[changeIndex])
            tmp_img = self.DrawSeq(tmp_seq)
            tmp_error = self.Evaluate(tmp_img)
            if tmp_error < cache_error:
                cache_error = tmp_error
                svg_seq[idx] = command[:]


    def MutateCommand(self, svg, blur_percent,prob_insert):
        svg_seq = svg.svg_seq
        tmp_img = self.Draw(svg)
        diff1 = cv2.subtract(self.img_grey, tmp_img) #values are too low
        diff2 = cv2.subtract(tmp_img,self.img_grey) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        img = np.copy(totalDiff)
        plt.imshow(img, cmap='gray')
        plt.show()
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #calculate blur level
        w = img.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
        #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0/mag.max()
        mag = mag*scale
        plt.imshow(mag, cmap='gray')
        plt.show()
        index = 0
        for i in range(len(svg_seq)-1):
            if random.random() < prob_insert:
                posX, posY = util_sample_from_img(mag)
                if random.random() < 0.5:
                    command = ['L', posX, posY]
                    svg_seq.insert(index+1, command)
                else:
                    curX = svg_seq[index][-2]
                    curY = svg_seq[index][-1]
                    command = ['C', curX + (posX - curX)/3,curY + (posY - curY)/3, 2*(posX - curX)/3,2*(posY - curY)/3, posX, posY]
                    svg_seq.insert(index+1, command)
                index += 1
            index += 1

    def Mutate(self, svg, blur_percent,prob_insert, seed):
        # self.MutateCommand(svg, blur_percent,prob_insert)
        self.MutatePos(svg, seed)


    def Upper_bound(self, pp, target):
        low, high = 0, len(pp)-1
        pos = len(pp)
        while low < high:
            mid = (low+high) // 2
            if pp[mid] <= target:
                low = mid+1
            else:
                high = mid
                pos = high
        if pp[low] > target:
            pos = low
        return pos

    def CrossOver(self,svg1, svg2):
        c = []
        svg_seq1 = svg1.svg_seq
        svg_seq2 = svg2.svg_seq
        len1 = len(svg_seq1)
        len2 = len(svg_seq2)
        min_len = min(len1, len2)
        max_len = max(len1, len2)
        if len1 < len2:
            min_seq = svg_seq1
            max_seq = svg_seq2
        else:
            min_seq = svg_seq2
            max_seq = svg_seq1
        for i in range(min_len // 2):
            c.append(min_seq[i])
        for i in range(min_len // 2, max_len):
            c.append(max_seq[i])
        new_svg = SVG(copy.deepcopy(c))
        return new_svg

    def Edit(self, generations, xi, decay, prob_crs, prob_mut):
        self.InitPopulation()
        p_best, p_worst = self.EvaluatePopulation()
        txi = xi
        for g in range(generations):
            clear_output(wait=True)
            print("Generation ", g+1, "/", generations)
            print(self.population[self.cur][p_best].loss)
            self.ComputeCrossProb(txi, self.population[self.cur][p_worst].loss)
            txi /= decay
            for i in range(self.pop_size):
                p1 = self.Upper_bound(self.pp, random.random())
                p2 = self.Upper_bound(self.pp, random.random())
                self.population[1 - self.cur][i] = copy.deepcopy(self.population[self.cur][p1])
                # if random.random() < prob_crs:
                #     self.population[1 - self.cur][i] = self.CrossOver(self.population[self.cur][p1],self.population[self.cur][p2])

            for i in range(self.pop_size):
                # if random.random() < prob_mut:
                    self.Mutate(self.population[1 - self.cur][i],0.05, 0.1 , self.seed + time.time() + g)

            # for i in range(1, self.pop_size + 1):
            #     tmp_img = self.Draw(self.population[1 - self.cur][i-1])
            #     plt.subplot(2, 5, i)
            #     plt.imshow(tmp_img, cmap='gray')
            # plt.show()

            self.cur = 1 - self.cur
            c_best, c_worst = self.EvaluatePopulation()
            if self.population[1 - self.cur][p_best].loss < self.population[self.cur][c_best].loss:
                self.population[self.cur][c_worst] = copy.deepcopy(self.population[1 - self.cur][p_best])
                p_best = c_worst
            else:
                p_best = c_best
            loss_worst = 0
            for i in range(self.pop_size):
                if self.population[self.cur][i].loss > loss_worst:
                    loss_worst = self.population[self.cur][i].loss
                    p_worst = i

        final_svg =  self.population[self.cur][p_best]
        final_img = self.Draw(final_svg)
        diff1 = cv2.subtract(self.img_grey, final_img) #values are too low
        diff2 = cv2.subtract(final_img,self.img_grey) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        return final_svg, final_img, totalDiff


def main():
    editer = Editer('target_image/I10.png','source_svg/I.svg', 10, seed=time.time())
    svg, img, totalDiff= editer.Edit(100, 20, 0.9, 0.8, 0.9)
    print(svg)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(totalDiff, cmap = 'gray')
    plt.show()

if __name__ == '__main__':
    main()
