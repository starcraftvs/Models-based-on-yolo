import numpy as np
import cv2
import math
import copy
# from data_prepare.hom_trans import solve_hom


def find_dimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:
        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
        hp_arr = np.array(hp, np.float32)
        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)
        if max_x is None or normal_pt[0, 0] > max_x:
            max_x = normal_pt[0, 0]
        if max_y is None or normal_pt[1, 0] > max_y:
            max_y = normal_pt[1, 0]
        if min_x is None or normal_pt[0, 0] < min_x:
            min_x = normal_pt[0, 0]
        if min_y is None or normal_pt[1, 0] < min_y:
            min_y = normal_pt[1, 0]
    min_x = min(0, min_x)
    min_y = min(0, min_y)
    return min_x, min_y, max_x, max_y

def get_shift(src_img, hom, debug=False):
    hom = copy.deepcopy(hom)
    h, w, _ = src_img.shape
    src_pts = [[0, 0], [0, h], [w, h], [w, 0]]
    pts = np.float32(src_pts).reshape((-1, 1, 2))
    dst_pts = cv2.perspectiveTransform(pts, hom)
    dst_pts = dst_pts.reshape((-1, 2))
    xmin, ymin = np.int32(dst_pts.min(axis=0).ravel())
    xmax, ymax = np.int32(dst_pts.max(axis=0).ravel())
    t = [-xmin, -ymin]
    ht = np.float32([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    if debug:
        print(ht)
    return ht, (xmin, xmax, ymin, ymax)

def gen_homography(width, height, angle, shift_v, shift_h, shear, f_global=28):
    # print(f"width:{width}\nheight:{height}\nangle:{angle}\nshift_v:{shift_v}\nshift_h:{shift_h}\nshear:{shear}")
    u = width
    v = height
    phi = math.pi * angle / 180.0
    cosi = math.cos(phi)
    sini = math.sin(phi)
    # ascale = 1.0

    exppa_v = math.exp(shift_v)
    # fdb_v = f_global / (14.4 + (v / u - 1) * 7.2)
    # rad_v = fdb_v * (exppa_v - 1.0) / (exppa_v + 1.0)
    # alpha_v = max(-1.5, min(1.5, math.atan(rad_v)))
    # rt_v = math.sin(0.5 * alpha_v)
    ## r_v = 1.0

    # vertifac = 1.0
    exppa_h = math.exp(shift_h)
    # fdb_h = f_global / (14.4 + (u / v - 1) * 7.2)
    # rad_h = fdb_h * (exppa_h - 1.0) / (exppa_h + 1.0)
    # alpha_h = max(-1.5, min(1.5, math.atan(rad_h)))
    # rt_h = math.sin(0.5 * alpha_h)
    ## r_h = 1.0

    # Step 1: flip x and y coordinates (see above)
    minput = np.zeros((3, 3))
    minput[0][1] = 1.0
    minput[1][0] = 1.0
    minput[2][2] = 1.0

    # Step 2: rotation of image around its center
    mwork = np.zeros((3, 3))
    mwork[0][0] = cosi
    mwork[0][1] = -sini
    mwork[1][0] = sini
    mwork[1][1] = cosi
    mwork[0][2] = -0.5 * v * cosi + 0.5 * u * sini + 0.5 * v
    mwork[1][2] = -0.5 * v * sini - 0.5 * u * cosi + 0.5 * u
    mwork[2][2] = 1.0

    moutput = mwork @ minput

    # Step 3: apply shearing
    mwork = np.zeros((3, 3))
    mwork[0][0] = 1.0
    mwork[0][1] = shear
    mwork[1][1] = 1.0
    mwork[1][0] = shear
    mwork[2][2] = 1.0
    moutput = mwork @ moutput

    # Step 4: apply vertical lens shift effect
    mwork = np.zeros((3, 3))
    mwork[0][0] = exppa_v
    mwork[1][0] = 0.5 * ((exppa_v - 1.0) * u) / v
    mwork[1][1] = 2.0 * exppa_v / (exppa_v + 1.0)
    mwork[1][2] = -0.5 * ((exppa_v - 1.0) * u) / (exppa_v + 1.0)
    mwork[2][0] = (exppa_v - 1.0) / v
    mwork[2][2] = 1.0
    moutput = mwork @ moutput

    # Step 5: horizontal compression
    # mwork = np.zeros((3, 3))
    # mwork[0][0] = 1.0
    # mwork[1][1] = r_v
    # mwork[1][2] = 0.5 * u * (1.0 - r_v)
    # mwork[2][2] = 1.0
    # moutput = mwork @ moutput

    # Step 6: flip x and y back again
    mwork = np.zeros((3, 3))
    mwork[0][1] = 1.0
    mwork[1][0] = 1.0
    mwork[2][2] = 1.0
    moutput = mwork @ moutput

    # from here output vectors would be in (x : y : 1) format

    # Step 7: now we can apply horizontal lens shift with the same matrix format as above
    mwork = np.zeros((3, 3))
    mwork[0][0] = exppa_h
    mwork[1][0] = 0.5 * ((exppa_h - 1.0) * v) / u
    mwork[1][1] = 2.0 * exppa_h / (exppa_h + 1.0)
    mwork[1][2] = -0.5 * ((exppa_h - 1.0) * v) / (exppa_h + 1.0)
    mwork[2][0] = (exppa_h - 1.0) / u
    mwork[2][2] = 1.0
    # print('*****exppa_h', exppa_h)
    # print('*****', mwork)
    moutput = mwork @ moutput

    # Step 8: vertical compression
    # mwork = np.zeros((3, 3))
    # mwork[0][0] = 1.0
    # mwork[1][1] = r_h
    # mwork[1][2] = 0.5 * v * (1.0 - r_h)
    # mwork[2][2] = 1.0
    # moutput = mwork @ moutput

    # Step 9: apply aspect ratio scaling
    # mwork = np.zeros((3, 3))
    # mwork[0][0] = 1.0 * ascale
    # mwork[1][1] = 1.0 / ascale
    # mwork[2][2] = 1.0
    # moutput = mwork @ moutput

    # Step 10: find x/y offsets and apply according correction so that
    # no negative coordinates occur in output vector
    umin, vmin = np.inf, np.inf
    # visit all four corners
    for y in (0, height - 1):
        for x in (0, width - 1):
            pi = (x, y, 1.0)
            po = moutput @ pi
            umin = min(umin, po[0] / po[2])
            vmin = min(vmin, po[1] / po[2])
    mwork = np.zeros((3, 3))
    mwork[0][0] = 1.0
    mwork[1][1] = 1.0
    mwork[2][2] = 1.0
    mwork[0][2] = -umin
    mwork[1][2] = -vmin
    moutput = mwork @ moutput

    return moutput