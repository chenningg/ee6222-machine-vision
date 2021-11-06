# AB - physical distance in mm between center points of two crosses. physical distance measured by measuring tape
# d - physical distance in mm from camera lens to center point between two crosses. physical distance measured by measuring tape
# h - distance in pixels between center points of two crosses in the image. pixel location determined by using MS Paint and reading off pixel coordinates

AB = 1.2 * 1000


# image 1: closer image : q1_img_1
OL_1 = 1.24 * 1000 
XA_1, XB_1 = 1111, 157

# image 2: farther image : q1_img_2
OL_2 = 1.80 * 1000
XA_2, XB_2 = 925, 273


# method 1
# using the formula found on slide 4 of lecture slides [2] and adapting it
# OO = ( (XA_1 - XB_1) / AB * OL_1 - (XA_2 - XB_2) / AB * OL_2)     /     ((XA_1 - XB_1) / AB - (XA_2 - XB_2) / AB)
OO = ( (XA_1 - XB_1) * OL_1 - (XA_2 - XB_2) * OL_2 )     /     ( (XA_1 - XB_1) - (XA_2 - XB_2) )
print("focal length: ", OO)


FX = (XA_1 - XB_1) / AB * OL_1 - (XA_1 - XB_1) / AB * OO
print("pixel focal length: ", FX)


# method 2  
# using the formula : h/H = f/L to find f
FX_alt1 = (XA_1 - XB_1) / AB * OL_1
FX_alt2 = (XA_2 - XB_2) / AB * OL_2

print("pixel_focal_length_1: ", FX_alt1)
print("pixel_focal_length_2: ", FX_alt2)