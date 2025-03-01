def get_bbox_center(bbox):
    """
        How (x1+x2)/2, (y1+y2)/2 ?
        - x_center = x1 + (x2-x1)/2 = ( 2x1 + x2 - x1 ) / 2 = (x1+x2)/2
    """

    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return x2-x1