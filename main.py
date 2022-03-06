import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from cairosvg import svg2png
import time
import matplotlib.pyplot as plt
import copy
import argparse
import random
import re
import os
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
        self.inital_length = None


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
                command.append(True)
                command.append('modify')
                svg_seq.append(command)
        self.inital_length = len(svg_seq)
        for i in range(self.pop_size):
            svg = SVG(copy.deepcopy(svg_seq))
            self.population[self.cur].append(svg)
        self.population[1 - self.cur] = [None for i in range(self.pop_size)]

    def Draw(self, svg):
        svg_seq = svg.svg_seq
        svg_str = ''
        for i in range(len(svg_seq)):
            for j in svg_seq[i][:-2]:
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
            for j in svg_seq[i][:-2]:
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
        totalDiff = totalDiff / (self.height * self.width)
        return totalDiff

    def TotalLoss(self, svg_seq):
        render_img = self.DrawSeq(svg_seq)
        totalDiff = self.Evaluate(render_img)
        totalLoss = 0.8 * totalDiff + 0.2 * len(svg_seq)
        return totalLoss

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
            if svg_seq[idx][-1] == 'fix':
                continue
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

    def InsertCommand(self, svg, blur_percent):
        svg_seq = svg.svg_seq
        tmp_img = self.Draw(svg)
        diff1 = cv2.subtract(self.img_grey, tmp_img) #values are too low
        diff2 = cv2.subtract(tmp_img,self.img_grey) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        img = np.copy(totalDiff)
        if not __debug__:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
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
        if not __debug__:
            plt.subplot(1, 2, 2)
            plt.imshow(mag, cmap='gray')
            plt.show()
        index = 0
        for i in range(0, len(svg_seq)):
            curX = svg_seq[index][-4]
            curY = svg_seq[index][-3]
            if mag[int(curY)][int(curX)] > 100:
                while(True):
                        posX, posY = util_sample_from_img(mag)
                        dis = ((posX - curX)**2 + (posY - curY)**2)**0.5
                        if dis < 10:
                            break
                if random.random() < 0.5:
                    command = ['L', posX, posY, False, 'modify']
                    svg_seq.insert(index+1, command)
                else:
                    command = ['C', curX + (posX - curX)/3,curY + (posY - curY)/3, curX + 2*(posX - curX)/3, curY + 2*(posY - curY)/3, posX, posY, False, 'modify']
                    svg_seq.insert(index+1, command)
                    if (index+1) != len(svg_seq)-1:
                        svg_seq[index + 2][-1] = 'involve_modify'
                index += 1
            elif mag[int(curY)][int(curX)] < 60:
                if svg_seq[index][-1] != 'involve_modify':
                    svg_seq[index][-1] = 'fix'
            index += 1

    # def DeleteCommand(self, svg):
    #     svg_seq = svg.svg_seq
    #     cache_totalloss = self.TotalLoss(svg_seq)
    #     for i in range(len(svg_seq)):
    #         tmp_seq = copy.deepcopy(svg_seq)
    #         if tmp_seq[i][-1] == False:
    #             del tmp_seq[i]
    #             totalloss = self.TotalLoss(tmp_seq)
    #             if totalloss < cache_totalloss:
    #                 del svg_seq[i]
    #                 cache_totalloss = totalloss

    def ModifyAll(self):
        for i in range(self.pop_size):
            for command in self.population[self.cur][i].svg_seq:
                command[-1] = 'modify'

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
        middle_len = self.inital_length // 2
        if random.random() < 0.5:
            first_seq = svg_seq1
            second_seq = svg_seq2
        else:
            first_seq = svg_seq2
            second_seq = svg_seq1

        len1 = len(first_seq)
        len2 = len(second_seq)
        index = 0
        num = 0
        while(True):
            if first_seq[index][-2] == False:
                c.append(first_seq[index])
            elif first_seq[index][-2] == True:
                c.append(first_seq[index])
                num += 1
            index += 1
            if num == middle_len:
                break
        index = 0
        num = 0
        while(True):
            if second_seq[index][-2] == True:
                num += 1
            index += 1
            if num == middle_len:
                break
        for i in range(index,len2):
            c.append(second_seq[i])
        new_svg = SVG(copy.deepcopy(c))
        return new_svg

    def Edit(self, generations, xi, decay, prob_crs):
        self.InitPopulation()
        p_best, p_worst = self.EvaluatePopulation()
        txi = xi
        for g in range(generations):
            clear_output(wait=True)
            print("Generation ", g+1, "/", generations)
            print(self.population[self.cur][p_best].loss)
            print(len(self.population[self.cur][p_best].svg_seq))
            self.ComputeCrossProb(txi, self.population[self.cur][p_worst].loss)
            txi /= decay
            if g == 380:
                self.ModifyAll()

            if g < 100 or (g > 210 and g < 250) or (g > 350 and g < 400):
                for i in range(self.pop_size):
                    p1 = self.Upper_bound(self.pp, random.random())
                    p2 = self.Upper_bound(self.pp, random.random())
                    self.population[1 - self.cur][i] = copy.deepcopy(self.population[self.cur][p1])
                    if random.random() < prob_crs:
                        self.population[1 - self.cur][i] = self.CrossOver(self.population[self.cur][p1],self.population[self.cur][p2])
            else:
                for i in range(self.pop_size):
                    self.population[1 - self.cur][i] = copy.deepcopy(self.population[self.cur][i])

            for i in range(self.pop_size):
                if g == 100 or g == 250:
                    if not __debug__:
                        tmp_img = self.Draw(self.population[1 - self.cur][i])
                    self.InsertCommand(self.population[1 - self.cur][i], 0.02)
                    if not __debug__:
                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(tmp_img, cmap='gray')
                        tmp_img = self.Draw(self.population[1 - self.cur][i])
                        plt.subplot(1,2,2)
                        plt.imshow(tmp_img, cmap='gray')
                        plt.show()
                self.MutatePos(self.population[1 - self.cur][i], self.seed + time.time() + g)

            self.cur = 1 - self.cur
            c_best, c_worst = self.EvaluatePopulation()

            if g < 100 or (g > 210 and g < 250) or (g > 350 and g < 400):
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
            else:
                p_best = c_best
                p_worst = c_worst

        final_svg =  self.population[self.cur][p_best]
        final_img = self.Draw(final_svg)
        diff1 = cv2.subtract(self.img_grey, final_img) #values are too low
        diff2 = cv2.subtract(final_img,self.img_grey) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        return final_svg, final_img, totalDiff

def save_svg(svg, target_dir, name):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    svg_seq = svg.svg_seq
    svg_str = ''
    for i in range(len(svg_seq)):
        for j in svg_seq[i][:-2]:
            svg_str += str(j)
            svg_str += ' '
    svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
    svg_data += '<path d="%s" stroke-width="1.0" fill="rgb(0, 0, 0)" opacity="1.0"/></g></svg>'%svg_str
    svg_outfile = os.path.join(target_dir, f"{name}.svg")
    svg_f = open(svg_outfile, 'w')
    svg_f.write(svg_data)
    svg_f.close()

def main():
    parser = argparse.ArgumentParser(description="LMDB creation")
    parser.add_argument('--char_class', type=str, default='B')
    opts = parser.parse_args()
    print(opts.char_class)
    editer = Editer(f'target_image/{opts.char_class}.png',f'source_svg/{opts.char_class}.svg', 10, seed=time.time())
    target_dir = 'target_svg'
    svg, img, totalDiff= editer.Edit(400, 20, 0.9, 0.8)
    if not __debug__:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap = 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(totalDiff, cmap = 'gray')
        plt.show()
    save_svg(svg, target_dir, opts.char_class)

if __name__ == '__main__':
    main()
