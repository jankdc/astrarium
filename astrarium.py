from __future__ import print_function

import cv2

def set_label(img, label, cnt):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    white_color = (255, 255, 255)
    filled = cv2.cv.CV_FILLED
    (text_w, text_h), baseline = cv2.getTextSize(label, fontface, scale, thickness)
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(cnt)
    pt0 = (rect_x + ((rect_w - text_w) / 2), rect_y + ((rect_h + text_h) / 2))
    pt1 = (pt0[0], pt0[1] + baseline)
    pt2 = (pt0[0] + text_w, pt0[1] - text_h)
    cv2.rectangle(img, pt1, pt2, white_color, filled)
    cv2.putText(img, label, pt0, fontface, scale, (0, 0, 0), thickness, 8)


def main():
    src = cv2.imread('data/shapes.png')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    binary = cv2.Canny(gray, 0, 50, apertureSize=5)
    cnts, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dst = src.copy()

    for cnt in cnts:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
        is_too_small = cv2.contourArea(cnt) < 100

        if is_too_small or not cv2.isContourConvex(cnt):
            continue

        if len(cnt) == 3:
            set_label(dst, 'TRI', cnt)


    cv2.imshow('destination', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
