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
        path = path_.split('" fill')[0]
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

        return svg_seq

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
            if delete_loss  <= current_loss + 0.25:
                return_svg_seq = copy.deepcopy(tmp_seq)
        return return_svg_seq

    def test_delete(self):
        svg_seq = self.InitPopulation()
        length = 0
        for command in svg_seq:
            if command[-2] != 'del_true' and command[-2] != 'del_false':
                length += 1
        print(length)
        del_seq = self.DeleteCommand(svg_seq)
        length = 0
        for command in del_seq:
            if command[-2] != 'del_true' and command[-2] != 'del_false':
                length += 1
        print(length)
        target_dir = 'target_svg_outlines'
        save_svg_outlines(del_seq, target_dir, 'D')


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
    editer = Editer(f'target_image/149_2/D.png',f'target_svg_outlines/149_2/D.svg', 10, seed=time.time())
    editer.test_delete()

if __name__ == '__main__':
    main()
