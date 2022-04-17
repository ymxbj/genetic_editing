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

class FloutRange:
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

def get_img_bbox(img):
    img = 255 - np.array(img)
    img0 = np.sum(img, axis = 0)
    img1 = np.sum(img, axis = 1)
    y_range = np.where(img1>127.5)[0]
    x_range = np.where(img0>127.5)[0]
    return [x_range[0],x_range[-1]], [y_range[0],y_range[-1]]

class Editer:
    def __init__(self,img_path,svg_path, pop_size, seed = 0):
        self.img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gx = cv2.Sobel(self.img_grey, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(self.img_grey, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # mag = cv2.bitwise_not(mag)
        mag = 255 - mag
        self.img_outline = mag.astype(np.uint8)
        self.svg_seq = None
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
                svg_seq.append(command)
        self.inital_length = len(svg_seq)
        self.svg_seq = svg_seq

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

    def Evaluate(self, render_img):
        diff_img_1 = cv2.subtract(self.img_grey, render_img) #values are too low
        diff_img_2 = cv2.subtract(render_img,self.img_grey) #values are too high
        totalDiff_img = cv2.add(diff_img_1, diff_img_2)
        totalDiff_img = np.sum(totalDiff_img)
        totalDiff_img = totalDiff_img / (self.height * self.width)

        # diff_outline_1 = cv2.subtract(self.img_outline, render_outline) #values are too low
        # diff_outline_2 = cv2.subtract(render_outline,self.img_outline) #values are too high
        # totalDiff_outline = cv2.add(diff_outline_1, diff_outline_2)
        # totalDiff_outline = np.sum(totalDiff_outline)
        # totalDiff_outline = totalDiff_outline / (self.height * self.width)

        totalDiff = totalDiff_img

        return totalDiff

    def Affine_transform(self, matrix):
        affined_svg = []
        for command in self.svg_seq:
            if command[0] == 'M' or command[0] == 'L':
                x = command[1]
                y = command[2]
                source_coord = np.array([[x],[y],[1]])
            elif command[0] == 'C':
                source_coord = np.array([[command[1],command[3],command[5]],[command[2],command[4],command[6]],[1,1,1]])
            target_coord = np.dot(matrix, source_coord)
            target_command = []
            target_command.append(command[0])
            if command[0] == 'M' or command[0] == 'L':
                target_command.append(target_coord[0][0])
                target_command.append(target_coord[1][0])
            elif command[0] == 'C':
                target_command.append(target_coord[0][0])
                target_command.append(target_coord[1][0])
                target_command.append(target_coord[0][1])
                target_command.append(target_coord[1][1])
                target_command.append(target_coord[0][2])
                target_command.append(target_coord[1][2])
            affined_svg.append(target_command)
        return affined_svg

    def fing_best_position(self):
        best_svg = None
        min_loss = 1000
        for scale_x in FloutRange(0.1,1,0.1):
            for scale_y in FloutRange(0.1,1,0.1):
                for shear_x in FloutRange(-0.5,0,0.05):
                    matrix = np.array([[scale_x, shear_x, 50],[0,scale_y, 50],[0,0,1]])
                    affined_svg = self.Affine_transform(matrix)
                    target_bbox = get_img_bbox(self.img_grey)
                    target_x = (target_bbox[0][0]+target_bbox[0][1])/2
                    target_y = (target_bbox[1][0]+target_bbox[1][1])/2
                    affined_img = self.DrawSeq(affined_svg)
                    # plt.figure()
                    # plt.imshow(affined_img, cmap='gray')
                    # plt.show()
                    affined_bbox = get_img_bbox(affined_img)
                    affined_x = (affined_bbox[0][0]+affined_bbox[0][1])/2
                    affined_y = (affined_bbox[1][0]+affined_bbox[1][1])/2
                    delta_x = target_x - affined_x
                    delta_y = target_y - affined_y
                    for command in affined_svg:
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
                    final_img = self.DrawSeq(affined_svg)
                    loss = self.Evaluate(final_img)
                    if loss <= min_loss:
                        min_loss = loss
                        best_svg = copy.deepcopy(affined_svg)
        return best_svg

    def test_affine_transform(self):
        self.InitPopulation()
        best_svg = self.fing_best_position()
        best_img = self.DrawSeq(best_svg)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(best_img, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(self.img_grey, cmap='gray')
        plt.show()



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

def save_svg_outlines(svg_seq, target_outlines_dir, name):
    if not os.path.exists(target_outlines_dir):
        os.mkdir(target_outlines_dir)
    svg_str = ''
    for i in range(len(svg_seq)):
        if svg_seq[i][-2] != 'del_true' and svg_seq[i][-2] != 'del_false':
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
    editer = Editer(f'target_image/149_2/G.png',f'source_svg/149/G.svg', 10, seed=time.time())
    editer.test_affine_transform()

if __name__ == '__main__':
    main()
