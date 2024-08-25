import os
os.chdir('C:\\Users\\karng\\OneDrive\\Desktop\\Object Tracking based on color\\Image Classification Using CNN\\Dataset\\test')
i = 1
for file in os.listdir():
    src = file
    dst = '0' + '_' + str(i) + '.jpg'
    os.rename(src,dst)
    i += 1