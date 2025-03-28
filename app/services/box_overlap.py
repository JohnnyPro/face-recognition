def boxes_overlap(boxA, boxB, overlap_threshold=0.2):
    """
    Check whether two bounding boxes overlap by a simple area ratio.
    Each box is (x_min, y_min, x_max, y_max).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    overlap = interArea / float(boxAArea)
    return overlap > overlap_threshold
