# 若要正常运行，则输入 python3 main.py --char_class I
# 其中char_class后的字符为要编辑的字母。
# 若要改为debug模式，即输出中间的状态图，则输入python3 -O main.py --char_class I
# 其中char_class后的字符为要编辑的字母。
for char in {A..Z}
do
    python3 main.py --char_class $char --font_class 179
done
