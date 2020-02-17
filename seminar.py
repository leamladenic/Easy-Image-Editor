import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy import stats
from PIL import Image, ImageOps
from PIL import ImageFilter


def gray_image(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R+G+B)/255
    grayImage = img

    for i in range(3):
        grayImage[:, :, i] = Avg
    cv2.imshow("siva slika", Avg)
    print(img.shape)
    print(grayImage.shape)
    print(Avg.shape)
    
def histogram(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    
def histogram_eq():
    photo = Image.open("slika.jpg")
    nova_slika = ImageOps.equalize(photo, mask=None)
    nova_slika.show("nova slika.jpg")

def histogram_norm():
    photo = Image.open("slika.jpg")
    nova = ImageOps.autocontrast(photo, cutoff=0)
    nova.show("nova slika.jpg")

def cumulative_hist(img):
    res = stats.cumfreq(img, numbins=25)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(img.ravel(), 256, [0, 256])
    ax1.set_title('Histogram')
    ax2.bar(x, res.cumcount, width=res.binsize)
    ax2.set_title('Cumulative histogram')
    plt.show()

def gamma_correction(img):
    image = img.astype(np.float32) / 255
    gamma = float(input("Please enter gamma: "))
    corrected_image = np.power(image, gamma)
    cv2.imshow('corrected_image', corrected_image)

def amp_segm(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = float(input("Please enter threshold: "))
    ret,thresh3 = cv2.threshold(grayImage,t,255,cv2.THRESH_TRUNC)
    cv2.imshow('corrected_image', thresh3)
    plt.subplot(2,1,1),plt.hist(img.ravel(), 256, [0, 256])
    plt.subplot(2,1,2),plt.imshow(thresh3,'gray')
    plt.show() 

def convolution(img):
    row = int(input("Please enter number of rows: "))
    col = int(input("Please enter number of columns: "))
    
    print("Please enter your filter: ")
    mat = []
    for i in range(row):
        a=[]
        for j in range(col):
            a.append(int(input()))
        mat.append(a) 

    #image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    my_filter=np.array(mat)
    flipVertical = np.flip(my_filter)
    print(flipVertical)
    filtrirana_slika = cv2.filter2D(img, -1, flipVertical, anchor=(-1,-1), borderType=cv2.BORDER_CONSTANT)
    cv2.imshow('corrected_image', filtrirana_slika)

def corellation(img):
    row = int(input("Please enter number of rows: "))
    col = int(input("Please enter number of columns: "))
    
    print("Please enter your filter: ")
    mat = []
    for i in range(row):
        a=[]
        for j in range(col):
            a.append(int(input()))
        mat.append(a) 
    
    #image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    my_filter=np.array(mat)
    print(my_filter)
    filtrirana_slika = cv2.filter2D(img, -1, my_filter, anchor=(-1,-1), borderType=cv2.BORDER_CONSTANT)
    cv2.imshow('corrected_image', filtrirana_slika)

def median_filter(img):
    retci = img.shape[0]
    stupci = img.shape[1]
    kanali = img.shape[2]
    filtrirana_slika = cv2.imread("slika.jpg")

    for k in range(0, kanali):
        for i in range(1, retci-1):
            for j in range(1, stupci-1):
                prazni_niz = np.zeros(9)
                brojac = 0
                for a in range(i-1,i+2):
                    for b in range(j-1,j+2):
                        piksel = img[a,b,k]
                        prazni_niz[brojac] = piksel
                        brojac = brojac + 1
              
                prazni_niz = np.sort(prazni_niz)
                
                filtrirana_slika[i,j,k] = prazni_niz[4]

    cv2.imshow('corrected_image', filtrirana_slika)

def img_sharp():
    imageObject = Image.open("slika.jpg")
    sharpened1 = imageObject.filter(ImageFilter.SHARPEN)
    sharpened1.show()

def mag_gradient(img):
    moj_filterX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    moj_filterY = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    filtrirana_slikaX = cv2.filter2D(img, -1, moj_filterX, anchor=(-1,-1), borderType=cv2.BORDER_CONSTANT)
    filtrirana_slikaY = cv2.filter2D(img, -1, moj_filterY, anchor=(-1,-1), borderType=cv2.BORDER_CONSTANT)

    filtrirana_slika=filtrirana_slikaX + filtrirana_slikaY

    cv2.imshow('corrected_image', filtrirana_slika)

def erozija(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (prag, slika_binarna) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    moj_filter = np.ones((5,5), np.uint8)

    slika_erozija = cv2.erode(slika_binarna, moj_filter, iterations = 1)

    cv2.imshow("Erozija", slika_erozija)

def dilatacija(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (prag, slika_binarna) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    moj_filter = np.ones((5,5), np.uint8)

    slika_dilatacija = cv2.dilate(slika_binarna, moj_filter, iterations = 1)

    cv2.imshow("Dilatacija", slika_dilatacija)

def opening(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (prag, slika_binarna) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    moj_filter = np.ones((5,5), np.uint8)

    slika_erozija = cv2.erode(slika_binarna, moj_filter, iterations = 1)
    slika_dilatacija = cv2.dilate(slika_erozija, moj_filter, iterations = 1)

    cv2.imshow("Opening", slika_erozija)

def closing(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (prag, slika_binarna) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    moj_filter = np.ones((5,5), np.uint8)

    slika_dilatacija = cv2.dilate(slika_binarna, moj_filter, iterations = 1)
    slika_erozija = cv2.erode(slika_dilatacija, moj_filter, iterations = 1)

    cv2.imshow("Closing", slika_erozija)

def morf_oper(img):
    print("1 - Erozija\n2 - Dilatacija\n3 - Opening\n4 - Closing\n")
    izbor2 = int(input("Please enter your choice: "))

    if izbor2 == 1:
        erozija(img)
    elif izbor2 == 2:
        dilatacija(img)
    elif izbor2 == 3:
        opening(img)
    else:
        closing(img)

def combine():
    image1 = cv2.imread("slika.jpg")
    image2 = cv2.imread("slika2.jpg")


    My_img = cv2.addWeighted(image1, 0.6, image2, 0.4, 0)
    
    cv2.imshow('image', My_img)
    

print("0 - Prikaz orginalne slike\n1 - Siva slika\n2 - Histogram\n3 - Kumulativni histogram\n4 - Ujednacavanje histograma\n5 - Rastezanje histograma\n6 - Gamma korekcija\n7 - Amplitudna segmentacija\n8 - Konvolucija\n9 - Korelacija\n10 - Median filter\n11 - Izostravanje\n12 - Magnituda gradijenta\n13 - Kombiniranje slika\n14 - Morfoloske operacije\n")
izbor = int(input("Please enter your choice: "))

photo = cv2.imread("slika.jpg")

if izbor == 0 :
    cv2.imshow("Slika", photo)
elif izbor == 1 :
    gray_image(photo)
elif izbor == 2 : 
    histogram(photo)
elif izbor == 3:
    cumulative_hist(photo)
elif izbor == 4 :
    histogram_eq()
elif izbor == 5 :
    histogram_norm()
elif izbor == 6 :
    gamma_correction(photo)
elif izbor == 7 :
    amp_segm(photo)
elif izbor == 8 :
    convolution(photo)
elif izbor == 9 :
    corellation(photo)
elif izbor == 10 :
    median_filter(photo)
elif izbor == 11 :
    img_sharp()
elif izbor == 12 : 
    mag_gradient(photo)
elif izbor == 13 :
    combine()
elif izbor == 14 : 
    morf_oper(photo)



cv2.waitKey(0)
cv2.destroyAllWindows()
