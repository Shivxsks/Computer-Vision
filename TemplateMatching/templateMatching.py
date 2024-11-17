import cv2
import numpy as np

original_image_template = cv2.imread(r'C:\Users\Admin\Downloads\TemplateMatching\images\4star.jpg')
cv2.imshow("Original Template", original_image_template)
cv2.waitKey(0)

height, width = original_image_template.shape[:2]

original_image = cv2.imread(r'C:\Users\Admin\Downloads\TemplateMatching\images\shapestomatch.jpg')
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)

gray_template = cv2.cvtColor(original_image_template, cv2.COLOR_BGR2GRAY)
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

match = cv2.matchTemplate(gray_original, gray_template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

top_left = max_loc
bottom_right = (top_left[0] + height, top_left[1] + width)
cv2.rectangle(original_image, top_left, bottom_right, (0, 0, 255), 5)

cv2.imshow("Original Image with Matched Area", original_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
