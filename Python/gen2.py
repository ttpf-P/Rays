import time

import numba
import numpy as np
import numpy.linalg as linalg
from PIL import Image

FRAMES = 12
X = 240  # height
Y = 240  # width

# bounded plane
# ("BPLANE", (unit normal vector), distance to origin, (maxVals), (minVals), (empty))

# plane
# ("PLANE", (unit normal vector), distance to origin, (empty), (empty), (empty))

# triangle
# ("TRIANGLE",(unit normal vector), distance to origin, (point0), (point1), (point2))

object_list = []


def bplane_from_points(point0=(0, 0, 0), point1=(0, 0, 1), point2=(1, 0, 1)):
    vector1 = np.array(point2) - np.array(point0)
    vector2 = np.array(point1) - np.array(point0)
    zipped_points = list(zip(point0, point1, point2))
    max_vals = (max(zipped_points[0]), max(zipped_points[1]), max(zipped_points[2]))
    min_vals = (min(zipped_points[0]), min(zipped_points[1]), min(zipped_points[2]))
    normal = np.cross(vector2, vector1)
    unit_normal = normal / linalg.norm(normal)
    dist_to_origin = sum(unit_normal * point0)
    return "BPLANE", unit_normal, dist_to_origin, max_vals, min_vals, (0, 0, 0)


def plane_from_points(point0=(0, 0, 0), point1=(0, 0, 1), point2=(1, 0, 1)):
    vector1 = np.array(point2) - np.array(point0)
    vector2 = np.array(point1) - np.array(point0)
    normal = np.cross(vector2, vector1)
    unit_normal = normal / linalg.norm(normal)
    dist_to_origin = sum(unit_normal * point0)
    return "PLANE", unit_normal, dist_to_origin, (0, 0, 0), (0, 0, 0), (0, 0, 0)


def triangle_from_points(point0=(0, 0, 0), point1=(0, 0, 1), point2=(1, 0, 1)):
    vector1 = np.array(point2) - np.array(point0)
    vector2 = np.array(point1) - np.array(point0)
    normal = np.cross(vector2, vector1)
    unit_normal = normal / linalg.norm(normal)
    dist_to_origin = sum(unit_normal * point0)
    return "TRIANGLE", unit_normal, dist_to_origin, point0, point1, point2


@numba.njit(cache=True)
def trace_ray(objects, origin=np.array((0, 0, 0), "float32"), direction=np.array((0, 1, 0), "float32")):
    direction = direction / linalg.norm(direction)  # normalize vector direction
    # print(direction)
    min_dist = -1
    for object_ in objects:
        if object_[0] == "PLANE":
            s = np.sum(object_[1] * direction)
            if s:
                dist = (object_[2] - (np.sum(object_[1] * origin))) / s
                if min_dist == -1 or min_dist > dist > 0:
                    min_dist = dist
        elif object_[0] == "BPLANE":
            s = np.sum(object_[1] * direction)
            # print(s)
            if s:  # bplane and ray are not parallel
                dist = (object_[2] - (np.sum(object_[1] * origin))) / s
                if min_dist == -1 or min_dist > dist > 0:
                    intersect = origin + (direction * dist)
                    if object_[3][0] * 1.001 >= intersect[0] >= object_[4][0] * 0.999 \
                            and object_[3][1] * 1.001 >= intersect[1] >= object_[4][1] * 0.999 \
                            and object_[3][2] * 1.001 >= intersect[2] >= object_[4][2] * 0.999 \
                            and dist > 0:
                        min_dist = dist
        elif object_[0] == "TRIANGLE":
            s = np.sum(object_[1] * direction)
            if s:  # ray intersects plane
                dist = (object_[2] - (np.sum(object_[1] * origin))) / s
                if min_dist == -1 or min_dist > dist > 0:  # intersect is nearest found
                    intersect = origin + (direction * dist)
                    # project onto plane
                    if object_[1][0] == max(object_[1]):
                        u0, u1, u2 = intersect[1] - object_[3][1], \
                                     object_[4][1] - object_[3][1], \
                                     object_[5][1] - object_[3][1]
                        v0, v1, v2 = intersect[2] - object_[3][2], \
                                     object_[4][2] - object_[3][2], \
                                     object_[5][2] - object_[3][2]
                    elif object_[1][1] == max(object_[1]):
                        u0, u1, u2 = intersect[0] - object_[3][0], \
                                     object_[4][0] - object_[3][0], \
                                     object_[5][0] - object_[3][0]
                        v0, v1, v2 = intersect[2] - object_[3][2], \
                                     object_[4][2] - object_[3][2], \
                                     object_[5][2] - object_[3][2]
                    else:
                        u0, u1, u2 = intersect[1] - object_[3][1], \
                                     object_[4][1] - object_[3][1], \
                                     object_[5][1] - object_[3][1]
                        v0, v1, v2 = intersect[0] - object_[3][0], \
                                     object_[4][0] - object_[3][0], \
                                     object_[5][0] - object_[3][0]

                    alpha = linalg.det(np.array(((u0, u2), (v0, v2)), "float32")) \
                        / linalg.det(np.array(((u1, u2), (v1, v2)), "float32"))
                    beta = linalg.det(np.array(((u1, u0), (v1, v0)), "float32")) \
                        / linalg.det(np.array(((u1, u2), (v1, v2)), "float32"))
                    if alpha >= 0 and beta >= 0 and alpha + beta <= 1:
                        min_dist = dist

    if min_dist != -1:
        return min_dist
    else:
        return 0


@numba.njit(parallel=False, cache=True)
def trace_rays(objects, X, Y, frame) -> np.array:
    pixels_raw = np.empty((X, Y))
    for x in numba.prange(X):
        # print("line")
        for y in range(Y):
            pixels_raw[x, y] = trace_ray(objects, origin=np.array((0, frame - 10, 0)),
                                         direction=np.array((2 * x / X - 1, 1, 2 * y / Y - 1)))
    return pixels_raw


object_list.append(("PLANE", np.array((0, -1, 0), "float64"), -4., (0, 0, 0), (0, 0, 0), (0, 0, 0)))
object_list.append(bplane_from_points((1, 2, 1), (1, 2, -1), (-1, 2, 1)))
object_list.append(triangle_from_points((-2, -1, 2), (3, -1, 7), (3, -3, -2)))

# print(object_list)
object_list = tuple(object_list)

print("compiling...")
trace_ray(object_list)
print("compiled: trace_ray")
trace_rays((bplane_from_points((1, 2, 1), (1, 2, -1), (-1, 2, 1)),), 2, 2, 0)
print("compiled: trace_rays")
print("finished compiling")

start = time.perf_counter_ns()
for frame in range(FRAMES):
    frame_start = time.perf_counter_ns()
    pixels = trace_rays(object_list, X, Y, frame)
    try:
        pixels = pixels * (255 / np.max(pixels))
    except ZeroDivisionError:
        pass

    im = Image.fromarray(pixels)
    im = im.convert("L")
    im.save("renders\\" + str(frame) + ".png", "png")
    print("rendered frame:", frame, "\ttime:", (time.perf_counter_ns() - frame_start) / 10 ** 9, "seconds")
print("rendering took:", (time.perf_counter_ns() - start) / 10 ** 9, "seconds")
