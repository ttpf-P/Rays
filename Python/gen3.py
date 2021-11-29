import numpy as np
import numpy.linalg as linalg
import scipy.linalg
from PIL import Image
import numba
import numba.cuda as cuda
import time
import trimesh.creation


def triangle_from_points(point0=(0, 0, 0), point1=(0, 0, 1), point2=(1, 0, 1)):
    vector1 = np.array(point2) - np.array(point0)
    vector2 = np.array(point1) - np.array(point0)
    normal = np.cross(vector2, vector1)
    unit_normal = normal / scipy.linalg.norm(normal)
    dist_to_origin = sum(unit_normal * point0)
    return np.array(unit_normal, "float32"), np.array((dist_to_origin, 0, 0), "float64"), \
        np.array(point0, "float64"), np.array(point1, "float64"), np.array(point2, "float64")


@numba.njit()
def trace_ray(objects, X, Y, x, y, frame):
    origin = np.array((0, -2, frame), "float64")
    direction = np.array((2 * x / X - 1, 1, (2 * y / Y - 1)-(frame/10)), "float64")
    direction = direction / linalg.norm(direction)
    min_dist = -1
    for object_ in objects:
        s = np.sum(object_[0] * direction)
        if s:  # ray intersects plane
            dist = (object_[1][0] - (np.sum(object_[0] * origin))) / s
            if min_dist == -1 or min_dist > dist > 0:  # intersect is nearest found
                intersect = origin + (direction * dist)
                # project onto plane
                if object_[0][0] == max(object_[0]):
                    u0, u1, u2 = intersect[1] - object_[2][1], \
                                 object_[3][1] - object_[2][1], \
                                 object_[4][1] - object_[2][1]
                    v0, v1, v2 = intersect[2] - object_[2][2], \
                                 object_[3][2] - object_[2][2], \
                                 object_[4][2] - object_[2][2]
                elif object_[0][1] == max(object_[0]):
                    u0, u1, u2 = intersect[0] - object_[2][0], \
                                 object_[3][0] - object_[2][0], \
                                 object_[4][0] - object_[2][0]
                    v0, v1, v2 = intersect[2] - object_[2][2], \
                                 object_[3][2] - object_[2][2], \
                                 object_[4][2] - object_[2][2]
                else:
                    u0, u1, u2 = intersect[1] - object_[2][1], \
                                 object_[3][1] - object_[2][1], \
                                 object_[4][1] - object_[2][1]
                    v0, v1, v2 = intersect[0] - object_[2][0], \
                                 object_[3][0] - object_[2][0], \
                                 object_[4][0] - object_[2][0]

                div0 = linalg.det(np.array(((u1, u2), (v1, v2)), "float64"))
                div1 = linalg.det(np.array(((u1, u2), (v1, v2)), "float64"))
                if div0 != 0 and div1 != 0:
                    alpha = linalg.det(np.array(((u0, u2), (v0, v2)), "float64")) \
                        / div0
                    beta = linalg.det(np.array(((u1, u0), (v1, v0)), "float64")) \
                        / div1
                    if alpha >= -0.001 and beta >= -0.001 and alpha + beta <= 1.001:  # account for float imprecision
                        min_dist = dist
                #else:
                #    alpha =.5
                #    beta = .5


    return min_dist


@cuda.jit()
def trace_ray_cuda(output, objects, X, Y, frame):
    z = cuda.grid(1)
    x = z // X
    y = z % X
    #print(z, x, y)
    origin = np.array((0, 0, 0), "float32")
    direction = np.array((2 * x / X - 1, 1, 2 * y / Y - 1), "float32")
    direction = direction / linalg.norm(direction)
    min_dist = -1
    for object_ in objects:
        s = np.sum(object_[0] * direction)
        if s:  # ray intersects plane
            dist = (object_[1][0] - (np.sum(object_[0] * origin))) / s
            if min_dist == -1 or min_dist > dist > 0:  # intersect is nearest found
                intersect = origin + (direction * dist)
                # project onto plane
                if object_[0][0] == max(object_[0]):
                    u0, u1, u2 = intersect[1] - object_[2][1], \
                                 object_[3][1] - object_[2][1], \
                                 object_[4][1] - object_[2][1]
                    v0, v1, v2 = intersect[2] - object_[2][2], \
                                 object_[3][2] - object_[2][2], \
                                 object_[4][2] - object_[2][2]
                elif object_[0][1] == max(object_[0]):
                    u0, u1, u2 = intersect[0] - object_[2][0], \
                                 object_[3][0] - object_[2][0], \
                                 object_[4][0] - object_[2][0]
                    v0, v1, v2 = intersect[2] - object_[2][2], \
                                 object_[3][2] - object_[2][2], \
                                 object_[4][2] - object_[2][2]
                else:
                    u0, u1, u2 = intersect[1] - object_[2][1], \
                                 object_[3][1] - object_[2][1], \
                                 object_[4][1] - object_[2][1]
                    v0, v1, v2 = intersect[0] - object_[2][0], \
                                 object_[3][0] - object_[2][0], \
                                 object_[4][0] - object_[2][0]
                div0 = linalg.det(np.array(((u1, u2), (v1, v2)), "float32"))
                div1 = linalg.det(np.array(((u1, u2), (v1, v2)), "float32"))
                if div0 != 0 and div1 != 0:
                    alpha = linalg.det(np.array(((u0, u2), (v0, v2)), "float32")) \
                            / div0
                    beta = linalg.det(np.array(((u1, u0), (v1, v0)), "float32")) \
                           / div1
                    if alpha >= 0 and beta >= 0 and alpha + beta <= 1:
                        min_dist = dist

    output[x, y] = min_dist


@numba.njit(parallel=True)
def trace_rays(X, Y, object_list, frame):
    output = np.empty((X, Y))
    for x in numba.prange(X):
        #print("line", x, "of", X, "in frame", frame)
        for y in range(Y):
            output[x][y] = trace_ray(object_list, X, Y, x, y, frame)

    return output
if __name__ == "__main__":
    X = 1080
    Y = 1080
    CUDA = False

    FRAMES = 1
    object_list = []

    #object_list.append(triangle_from_points((2, 2, 2), (2, 2, -2), (-2, 2, 2)))
    #object_list.append(triangle_from_points((-2, 4, -2), (-2, 4, 2), (2, 4, -2)))

    print("generating")
    sphere = trimesh.creation.icosphere(subdivisions=3)
    print(len(sphere.triangles))
    for triangle in sphere.triangles:
        object_list.append(triangle_from_points(*triangle))
    del sphere
    print("generated")

    object_list = np.array(object_list)

    # compile
    print("compiling")
    trace_ray(object_list, X, Y, 0, 0, 0)
    print("finished compiling")

    output = np.empty((X, Y))
    print(output.size)
    """for x in range(X):
        for y in range(Y):
            output[x][y] = trace_ray(object_list, X, Y, x, y, frame)"""

    if CUDA:
        threadsperblock = 40
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        for frame in range(FRAMES):
            frame_start = time.perf_counter_ns()
            output = np.empty((X, Y))

            trace_ray_cuda[blockspergrid, threadsperblock](output, object_list, X, Y, frame)

            output = output * (255 / np.max(output))

            im = Image.fromarray(output)
            im = im.convert("L")
            im.save("renders\\" + str(frame) + ".png", "png")
            print("rendered frame:", frame, "\ttime:", (time.perf_counter_ns() - frame_start) / 10 ** 9, "seconds")
    else:
        for frame in range(FRAMES):
            frame_start = time.perf_counter_ns()

            output = trace_rays(X, Y, object_list, frame)

            output = output * (255 / np.max(output))

            im = Image.fromarray(output)
            im = im.convert("L")
            im.save("renders\\" + str(frame) + ".png", "png")
            print("rendered frame:", frame, "\ttime:", (time.perf_counter_ns() - frame_start) / 10 ** 9, "seconds")
    print(trace_rays.parallel_diagnostics(level=4))
