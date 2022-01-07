
# Python program to rename all file
# names in your directory
import os

os.chdir('/home/kritanjali/Desktop/Internship/Clothing_Detection_YOLO/tests')
print(os.getcwd())

for count, f in enumerate(os.listdir()):
	f_name, f_ext = os.path.splitext(f)
	f_name = str(count)
	new_name = f'{f_name}'+".png"
	os.rename(f, new_name)
