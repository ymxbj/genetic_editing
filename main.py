import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from cairosvg import svg2png
import torch
from vgg_loss import VGGContextualLoss, VGGPerceptualLoss
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import argparse
import random
import re
import os
from IPython.display import clear_output

def showImage(img):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()

def showTwoImages(img1,img2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(img2, cmap='gray')
    plt.show()

class FloatRange:
    def __init__(self,start, end, step):
        self.start = start
        self.end = end
        self.step = step
        self.val = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.val > self.end:
            raise StopIteration
        val = self.val
        self.val += self.step
        return val

class SVG:
    def __init__(self,svg_seq):
        self.svg_seq = svg_seq
        self.loss = None

def get_img_bbox(img):
    img = 255 - np.array(img)
    img0 = np.sum(img, axis = 0)
    img1 = np.sum(img, axis = 1)
    y_range = np.where(img1 > 127.5)[0]
    x_range = np.where(img0 > 127.5)[0]
    return [x_range[0],x_range[-1]], [y_range[0],y_range[-1]]

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
        gx = cv2.Sobel(self.img_grey, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(self.img_grey, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # mag = cv2.bitwise_not(mag)
        mag = 255 - mag
        self.img_outline = mag.astype(np.uint8)
        self.svg_path = svg_path
        self.svg_seq = None
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

    def InitSeq(self):
        fin = open(self.svg_path,'r')
        path_ = fin.read().split('d="')[1]
        path = path_.split('" stroke')[0]
        path_splited = re.split(r"([MLC])", path)
        path_splited = path_splited[1:]

        svg_seq = []
        for i in range(len(path_splited)):
            if i%2 == 0:
                command = []
                command.append(path_splited[i])
            elif i%2 == 1:
                arg = path_splited[i].split()
                for j in range(len(arg)):
                    command.append(eval(arg[j]))
                svg_seq.append(command)

        self.svg_seq = svg_seq

    def Affine_transform(self, matrix):
        affined_svg_seq = []
        for command in self.svg_seq:
            if command[0] == 'M' or command[0] == 'L':
                source_coord = np.array([[command[1]],[command[2]],[1]])
            elif command[0] == 'C':
                source_coord = np.array([[command[1],command[3],command[5]],[command[2],command[4],command[6]],[1,1,1]])
            affined_coord = np.dot(matrix, source_coord)
            affined_command = []
            affined_command.append(command[0])
            if command[0] == 'M' or command[0] == 'L':
                affined_command.append(affined_coord[0][0])
                affined_command.append(affined_coord[1][0])
            elif command[0] == 'C':
                affined_command.append(affined_coord[0][0])
                affined_command.append(affined_coord[1][0])
                affined_command.append(affined_coord[0][1])
                affined_command.append(affined_coord[1][1])
                affined_command.append(affined_coord[0][2])
                affined_command.append(affined_coord[1][2])
            affined_command.append(True)
            affined_command.append('modify')
            affined_svg_seq.append(affined_command)
        return affined_svg_seq

    def move_to_canvas(self, affined_svg_seq):
        min_x = 1000
        max_x = -1000
        min_y = 1000
        max_y = -1000
        for command in affined_svg_seq:
            if command[0] == 'M' or command[0] == 'L':
                min_x = min(min_x, command[1])
                max_x = max(max_x, command[1])
                min_y = min(min_y, command[2])
                max_y = max(max_y, command[2])
            elif command[0] == 'C':
                min_x = min(min_x, command[1])
                max_x = max(max_x, command[1])
                min_y = min(min_y, command[2])
                max_y = max(max_y, command[2])
                min_x = min(min_x, command[3])
                max_x = max(max_x, command[3])
                min_y = min(min_y, command[4])
                max_y = max(max_y, command[4])
                min_x = min(min_x, command[5])
                max_x = max(max_x, command[5])
                min_y = min(min_y, command[6])
                max_y = max(max_y, command[6])

        if (max_x - min_x >= self.width) or (max_y - min_y >= self.height):
            return False
        else:
            center_x = (max_x + min_x)/2
            center_y = (max_y + min_y)/2
            delta_x = 128 - center_x
            delta_y = 128 - center_y

        for command in affined_svg_seq:
            if command[0] == 'M' or command[0] == 'L':
                command[1] += delta_x
                command[2] += delta_y
            elif command[0] == 'C':
                command[1] += delta_x
                command[2] += delta_y
                command[3] += delta_x
                command[4] += delta_y
                command[5] += delta_x
                command[6] += delta_y

        return True

    def Find_best_position(self):
        # contextual_loss = VGGContextualLoss(self.img_grey)
        perceptual_loss = VGGPerceptualLoss(self.img_grey)
        best_svg_seq = None
        min_loss = 1000
        target_bbox = get_img_bbox(self.img_grey)
        target_center_x = (target_bbox[0][0]+target_bbox[0][1])/2
        target_center_y = (target_bbox[1][0]+target_bbox[1][1])/2
        for scale_x in FloatRange(0.1,2,0.1):
            for scale_y in FloatRange(0.1,2,0.1):
                for shear_x in FloatRange(-0.3,0.3,0.1):
                    # scale_x = 1
                    # scale_y = 1
                    # shear_x = 0.2
                    matrix = np.array([[scale_x, shear_x, 0],[0,scale_y, 0],[0,0,1]])
                    affined_svg_seq = self.Affine_transform(matrix)
                    can_move = self.move_to_canvas(affined_svg_seq)
                    if can_move == False:
                        continue
                    affined_img = self.DrawSeq(affined_svg_seq)
                    affined_bbox = get_img_bbox(affined_img)
                    affined_center_x = (affined_bbox[0][0]+affined_bbox[0][1])/2
                    affined_center_y = (affined_bbox[1][0]+affined_bbox[1][1])/2
                    delta_x = target_center_x - affined_center_x
                    delta_y = target_center_y - affined_center_y
                    for command in affined_svg_seq:
                        if command[0] == 'M' or command[0] == 'L':
                            command[1] += delta_x
                            command[2] += delta_y
                        elif command[0] == 'C':
                            command[1] += delta_x
                            command[2] += delta_y
                            command[3] += delta_x
                            command[4] += delta_y
                            command[5] += delta_x
                            command[6] += delta_y
                    final_img = self.DrawSeq(affined_svg_seq)
                    # current_loss = contextual_loss(final_img, self.img_grey)
                    current_loss = perceptual_loss(final_img)
                    # print(current_loss)
                    # showTwoImages(final_img, self.img_grey)
                    if current_loss <= min_loss:
                        min_loss = current_loss
                        best_svg_seq = copy.deepcopy(affined_svg_seq)
        return best_svg_seq

    def InitPopulation(self):
        self.InitSeq()
        svg_seq = self.Find_best_position()
        print(len(svg_seq))
        if not __debug__:
            init_img = self.DrawSeq(svg_seq)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(init_img, cmap='gray')
        svg_seq = self.InitDeleteCommand(svg_seq)
        print(len(svg_seq))
        if not __debug__:
            plt.subplot(1,2,2)
            after_img = self.DrawSeq(svg_seq)
            plt.imshow(after_img, cmap='gray')
            plt.show()
        self.inital_length = len(svg_seq)
        if not __debug__:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(self.DrawSeqOutline(svg_seq), cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(self.img_outline, cmap='gray')
            plt.show()

        for i in range(self.pop_size):
            svg = SVG(copy.deepcopy(svg_seq))
            self.population[self.cur].append(svg)

        self.population[1 - self.cur] = [None for i in range(self.pop_size)]

    def Draw(self, svg):
        svg_seq = svg.svg_seq
        svg_str = ''
        for i in range(len(svg_seq)):
            if svg_seq[i][-2] != 'del_true' and svg_seq[i][-2] != 'del_false':
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

    def DrawOutline(self, svg):
        svg_seq = svg.svg_seq
        svg_str = ''
        for i in range(len(svg_seq)):
            if svg_seq[i][-2] != 'del_true' and svg_seq[i][-2] != 'del_false':
                for j in svg_seq[i][:-2]:
                    svg_str += str(j)
                    svg_str += ' '
        svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
        svg_data += '<path d="%s" fill="none" stroke="black" stroke-width="2.0"/></g></svg>'%svg_str
        png = svg2png(bytestring=svg_data)
        pil_img = Image.open(BytesIO(png))
        render_img = np.array(pil_img)
        render_img = render_img[:,:,-1]
        render_img = cv2.bitwise_not(render_img)
        return render_img

    def DrawSeq(self, svg_seq):
        svg_str = ''
        for i in range(len(svg_seq)):
            if svg_seq[i][-2] != 'del_true' and svg_seq[i][-2] != 'del_false':
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

    def DrawSeqOutline(self, svg_seq):
        svg_str = ''
        for i in range(len(svg_seq)):
            if svg_seq[i][-2] != 'del_true' and svg_seq[i][-2] != 'del_false':
                for j in svg_seq[i][:-2]:
                    svg_str += str(j)
                    svg_str += ' '
        svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
        svg_data += '<path d="%s" fill="none" stroke="black" stroke-width="2.0"/></g></svg>'%svg_str
        png = svg2png(bytestring=svg_data)
        pil_img = Image.open(BytesIO(png))
        render_outline = np.array(pil_img)
        render_outline = render_outline[:,:,-1]
        render_outline = cv2.bitwise_not(render_outline)
        return render_outline

    def Evaluate(self, render_img, render_outline):
        diff_img_1 = cv2.subtract(self.img_grey, render_img) #values are too low
        diff_img_2 = cv2.subtract(render_img,self.img_grey) #values are too high
        totalDiff_img = cv2.add(diff_img_1, diff_img_2)
        totalDiff_img = np.sum(totalDiff_img)
        totalDiff_img = totalDiff_img / (self.height * self.width)

        diff_outline_1 = cv2.subtract(self.img_outline, render_outline) #values are too low
        diff_outline_2 = cv2.subtract(render_outline,self.img_outline) #values are too high
        totalDiff_outline = cv2.add(diff_outline_1, diff_outline_2)
        totalDiff_outline = np.sum(totalDiff_outline)
        totalDiff_outline = totalDiff_outline / (self.height * self.width)

        totalDiff = totalDiff_img + 0.5 * totalDiff_outline

        return totalDiff

    def Deltaloss(self,svg_seq, tmp_seq, k):
        init_img = self.DrawSeq(svg_seq)
        cur_img = self.DrawSeq(tmp_seq)
        diff1 = cv2.subtract(init_img, cur_img) #values are too low
        diff2 = cv2.subtract(cur_img, init_img) #values are too high
        imgDiff = cv2.add(diff1, diff2)
        imgDiff = np.sum(imgDiff)
        imgDiff = imgDiff / (self.height * self.width)
        lengthDiff = len(svg_seq) - len(tmp_seq)
        deltaloss = lengthDiff - k * imgDiff
        return deltaloss

    def EvaluatePopulation(self):
        loss_best = 1e12
        loss_worst = 0
        for i in range(self.pop_size):
            svg = self.population[self.cur][i]
            render_img = self.Draw(svg)
            render_outline = self.DrawOutline(svg)
            svg.loss = self.Evaluate(render_img, render_outline)
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

    def MutatePos(self, svg, moving_dis, seed):
        svg_seq = svg.svg_seq
        cache_image = self.Draw(svg)
        cache_outline = self.DrawOutline(svg)
        cache_error = self.Evaluate(cache_image, cache_outline)
        for idx in range(len(svg_seq)):
            if svg_seq[idx][-1] == 'fix' or svg_seq[idx][-2] == 'del_true' or svg_seq[idx][-2] == 'del_false':
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
                delta = random.uniform(-moving_dis, moving_dis)
                command[changeIndex] += delta
                command[changeIndex] = self.Clamp(0, 256-1, command[changeIndex])
            tmp_img = self.DrawSeq(tmp_seq)
            tmp_outline = self.DrawSeqOutline(tmp_seq)
            tmp_error = self.Evaluate(tmp_img, tmp_outline)
            if tmp_error < cache_error:
                cache_error = tmp_error
                svg_seq[idx] = command[:]

    def CheckNeedInsert(self, mag, curX, curY):
        min_x = self.Clamp(0, self.width, curX - 5)
        max_x = self.Clamp(0, self.width, curX + 5)
        min_y = self.Clamp(0, self.height, curY - 5)
        max_y = self.Clamp(0, self.height, curY + 5)
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if mag[y][x] >= 100:
                    return True
        return False

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
            if svg_seq[index][-2] == 'del_true' or svg_seq[index][-2] == 'del_false':
                index += 1
                continue
            curX = svg_seq[index][-4]
            curY = svg_seq[index][-3]
            if self.CheckNeedInsert(mag, int(curX), int(curY)):
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
                    svg_seq.insert(index + 1, command)
                if (index + 1) != len(svg_seq) - 1:
                    tmp_index = index + 2
                    while svg_seq[tmp_index][-2] == 'del_true' or svg_seq[tmp_index][-2] == 'del_false':
                        tmp_index += 1
                        if tmp_index == len(svg_seq):
                            break
                    if tmp_index != len(svg_seq):
                        svg_seq[tmp_index][-1] = 'involve_modify'
                index += 1
            elif mag[int(curY)][int(curX)] < 60:
                if svg_seq[index][-1] != 'involve_modify':
                    svg_seq[index][-1] = 'fix'
            index += 1

    def InitDeleteCommand(self, svg_seq):
        return_svg_seq = copy.deepcopy(svg_seq)
        index = 2
        for i in range(2, len(return_svg_seq)):
            tmp_seq = copy.deepcopy(return_svg_seq)
            delete_command = tmp_seq[index]
            last_command = tmp_seq[index - 1]
            # last_last_command = tmp_seq[i-2]
            curX = delete_command[-4]
            curY = delete_command[-3]
            del tmp_seq[index]
            if last_command[0] == 'L':
                last_command[1] = curX
                last_command[2] = curY
            elif last_command[0] == 'C':
                last_command[-4] = curX
                last_command[-3] = curY
            deltaloss = self.Deltaloss(return_svg_seq, tmp_seq, 5)
            if deltaloss > 0:
                return_svg_seq = copy.deepcopy(tmp_seq)
            else:
                index += 1
        return return_svg_seq

    def DeleteCommand(self, svg_seq):
        return_svg_seq = copy.deepcopy(svg_seq)
        for i in range(1, len(return_svg_seq)):
            tmp_seq = copy.deepcopy(return_svg_seq)
            delete_command = tmp_seq[i]
            if delete_command[-2] == 'del_true' or delete_command[-2] == 'del_false' or delete_command[0] == 'M':
                continue
            index = i - 1
            last_command = tmp_seq[index]
            while(last_command[-2]=='del_true' or last_command[-2]=='del_false'):
                index -= 1
                last_command = tmp_seq[index]
            curX = delete_command[-4]
            curY = delete_command[-3]
            if delete_command[-2] == True:
                delete_command[-2] = 'del_true'
            elif delete_command[-2] == False:
                delete_command[-2] = 'del_false'
            last_command[-4] = curX
            last_command[-3] = curY
            cur_render_img = self.DrawSeq(return_svg_seq)
            cur_render_outlines = self.DrawSeqOutline(return_svg_seq)
            current_loss = self.Evaluate(cur_render_img, cur_render_outlines)
            del_render_img = self.DrawSeq(tmp_seq)
            del_render_outlines = self.DrawSeqOutline(tmp_seq)
            delete_loss = self.Evaluate(del_render_img, del_render_outlines)
            if delete_loss  <= current_loss + 0.15:
                return_svg_seq = copy.deepcopy(tmp_seq)
        return return_svg_seq

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
        middle_len = random.randint(2, self.inital_length - 1)
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
            elif first_seq[index][-2] == True or first_seq[index][-2] == 'del_true':
                c.append(first_seq[index])
                num += 1
            index += 1
            if num == middle_len:
                break
        index = 0
        num = 0
        while(True):
            if second_seq[index][-2] == True or second_seq[index][-2] == 'del_true':
                num += 1
            index += 1
            if num == middle_len:
                break
        for i in range(index,len2):
            if second_seq[i][-2] != 'del_false':
                c.append(second_seq[i])
        new_svg = SVG(copy.deepcopy(c))
        return new_svg

    def Edit(self, generations, xi, decay, prob_crs):
        self.InitPopulation()
        p_best, p_worst = self.EvaluatePopulation()
        txi = xi
        target_outlines_dir = f'target_svg_outlines/149_2'
        for g in range(generations):
            clear_output(wait=True)
            print("Generation ", g+1, "/", generations)
            print(self.population[self.cur][p_best].loss)
            length = 0
            for command in self.population[self.cur][p_best].svg_seq:
                if command[-2] != 'del_true' and command[-2] != 'del_false':
                    length += 1
            print(length)
            self.ComputeCrossProb(txi, self.population[self.cur][p_worst].loss)
            moving_dis = 2
            decay_rate = 0.9
            if g >= 400:
                moving_dis = moving_dis * np.power(decay_rate, (g-400) // 10 )
            if g == 380:
                self.ModifyAll()

            # if not __debug__:
            #     if g % 50 == 0:
            #         tmp_img = self.DrawOutline(self.population[self.cur][p_best])
            #         showImage(tmp_img)

            if g < 100 or (g > 210 and g < 250) or (g > 350 and g < generations):
                txi *= decay
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
                if g == 80 or g == 220 or g == 400:
                    self.population[1 - self.cur][i].svg_seq = copy.deepcopy(self.DeleteCommand(self.population[1 - self.cur][i].svg_seq))
                self.MutatePos(self.population[1 - self.cur][i],moving_dis, self.seed + time.time() + g)

            self.cur = 1 - self.cur
            c_best, c_worst = self.EvaluatePopulation()

            if g == 80 or g == 220 or g == 400:
                p_best = c_best
                p_worst = c_worst
            elif g < 100 or (g > 210 and g < 250) or (g > 350 and g < generations):
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

        redundant_svg =  self.population[self.cur][p_best]
        final_svg = delete_svg(redundant_svg)
        repair_svg(final_svg)
        final_img = self.Draw(final_svg)
        diff1 = cv2.subtract(self.img_grey, final_img) #values are too low
        diff2 = cv2.subtract(final_img,self.img_grey) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        return final_svg, final_img, totalDiff

def delete_svg(redundant_svg):
    svg_seq = redundant_svg.svg_seq
    final_svg_seq = []
    for command in svg_seq:
        if command[-2] != 'del_true' and command[-2] != 'del_false':
            final_svg_seq.append(command)
    final_svg = SVG(copy.deepcopy(final_svg_seq))
    return final_svg

def repair_svg(svg):
    svg_seq = svg.svg_seq
    startpoint_x = svg_seq[0][1]
    startpoint_y = svg_seq[0][2]
    index = 1
    for i in range(1, len(svg_seq)):
        if svg_seq[index][0] == 'M':
            command = ['L', startpoint_x, startpoint_y, True, 'modify']
            startpoint_x = svg_seq[index][1]
            startpoint_y = svg_seq[index][2]
            svg_seq.insert(index, command)
            index += 2
        else:
            index += 1
    command = ['L', startpoint_x, startpoint_y, True, 'modify']
    svg_seq.insert(len(svg_seq), command)

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

def save_svg_outlines(svg, target_outlines_dir, name):
    if not os.path.exists(target_outlines_dir):
        os.mkdir(target_outlines_dir)
    svg_seq = svg.svg_seq
    svg_str = ''
    for i in range(len(svg_seq)):
        for j in svg_seq[i][:-2]:
            svg_str += str(j)
            svg_str += ' '
    svg_data = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256"><defs/><g>'
    svg_data += '<path d="%s" fill="none" stroke="black" stroke-width="2.0"/></g></svg>'%svg_str
    svg_outfile = os.path.join(target_outlines_dir, f"{name}.svg")
    svg_f = open(svg_outfile, 'w')
    svg_f.write(svg_data)
    svg_f.close()

def main():
    parser = argparse.ArgumentParser(description="svg genetic editing")
    parser.add_argument('--char_class', type=str, default='A')
    parser.add_argument('--font_class', type=str, default='149')
    opts = parser.parse_args()
    print(opts.char_class)
    editer = Editer(f'target_image/149_4/{opts.char_class}.png',f'source_svg/{opts.font_class}/{opts.char_class}.svg', 10, seed=time.time())
    target_dir = f'target_svg/149_4'
    target_outlines_dir = f'target_svg_outlines/149_4'
    svg, img, totalDiff= editer.Edit(500, 20, 0.9, 0.8)
    if not __debug__:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap = 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(totalDiff, cmap = 'gray')
        plt.show()
    save_svg(svg, target_dir, opts.char_class)
    save_svg_outlines(svg, target_outlines_dir, opts.char_class)

if __name__ == '__main__':
    main()
