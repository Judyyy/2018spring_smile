import cv2
img_path = 'book.jpg'
image = cv2.imread(filename = img_path)
print (image)
cv2.imshow('book',image)
cv2.waitKey()
cv2.destroyAllWindows()