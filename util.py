import cv2 as cv


def add_text(image, text, org, font, font_scale, color, font_thickness=1):
    """
    Function to parse input string and label multi-lines text onto the image.
    """
    img_h, img_w = image.shape[:2]

    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, font_thickness)
    line_height = text_height + baseline

    for i, text in enumerate(text.split('\n')):
        y = org[1] + i*line_height
        cv.putText(image, text, (org[0], y), font, font_scale, color)
