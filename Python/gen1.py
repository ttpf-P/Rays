import numpy as np
import numpy.linalg
from PIL import Image


class vector:
    def __init__(self, origin=(0, 0, 0), direction=(1, 0, 0)):
        direction = np.array(direction)
        self.origin = np.array(origin)
        self.direction = direction / np.linalg.norm(direction)
        #print("dir:", self.direction)


class plane:
    def __init__(self, points=((0, 2, 1), (0, 2, 2), (1, 2, 1))):
        vector1 = np.array(points[0]) - np.array(points[1])
        vector2 = np.array(points[1]) - np.array(points[2])
        normal = np.cross(vector1, vector2)
        print("norm", normal)
        self.origin = sum(normal * points[0])  # d
        self.mag = numpy.linalg.norm(normal)
        self.normal = normal / self.mag  # unit normal vector
        print("norm", self.normal)
        print("origin", self.origin)

    def check_intersection(self, vector):
        s = sum(self.normal * vector.direction)
        #print("s:", s)
        if s:
            dist = (self.origin - (sum(self.normal * vector.origin))) / s
            print("dist:", dist)
            intersect = vector.origin + (vector.direction * dist)
            print("intersect:", intersect)
            return dist
        return False


class bounded_plane:
    def __init__(self, points=((0, 2, 1), (0, 2, 2), (1, 2, 1))):
        vector1 = np.array(points[2]) - np.array(points[0])
        vector2 = np.array(points[1]) - np.array(points[0])
        zipped_points = list(zip(*points))

        self.maxVals = [max(zipped_points[0]), max(zipped_points[1]), max(zipped_points[2])]
        self.minVals = [min(zipped_points[0]), min(zipped_points[1]), min(zipped_points[2])]
        print("minmax", self.minVals, self.maxVals)

        normal = np.cross(vector2, vector1)
        #print("norm", normal)
        self.mag = numpy.linalg.norm(normal)
        self.normal = normal / self.mag  # unit normal vector
        self.origin = sum(self.normal * points[0])  # d
        #print("norm", self.normal)
        #print("origin", self.origin)

    def check_intersection(self, vector):
        s = sum(self.normal * vector.direction)
        #print("s:", s)
        if s:
            dist = (self.origin - (sum(self.normal * vector.origin))) / s
            #print("dist:", dist)
            intersect = vector.origin + (vector.direction * dist)
            #print("intersect:", intersect)
            if self.maxVals[0]*1.001 >= intersect[0] >= self.minVals[0]*0.999 \
                    and self.maxVals[1]*1.001 >= intersect[1] >= self.minVals[1]*0.999 \
                    and self.maxVals[2]*1.001 >= intersect[2] >= self.minVals[2]*0.999 \
                    and dist > 0:
                #print(dist)
                return dist
        #raise
        return 0

if __name__ == "__main__":

    FRAMES = 1
    X = 240
    Y = 240
    OVERRENDER = 1


    v = vector((0, 0, 0), (0, 1, 0))
    p = plane()
    print(p.check_intersection(v))
    b = bounded_plane(((60, 5, 60), (180, 5, 60), (180, 5, 180)))
    print(b.check_intersection(v))


    for frame in range(FRAMES):
        pixels_raw = []
        for x in range(X*OVERRENDER):
            pixels_raw.append([])
            for y in range(Y*OVERRENDER):
                v = vector((120, 10*frame-100, 120), (2*x/(X*OVERRENDER)-1, 1, 2*y/(Y*OVERRENDER)-1))
                pixels_raw[-1].append(b.check_intersection(v))
        pixels_raw = np.array(pixels_raw)
        pixels_raw = pixels_raw*(255/np.max(pixels_raw))

        if OVERRENDER != 1:
            pixels = []
            for x in range(X-1):
                pixels.append([])
                for y in range(Y-1):
                    pixel = [[pixels_raw[x_][y_] for x_ in range(x*OVERRENDER, (x+1)*OVERRENDER)]
                                          for y_ in range(y*OVERRENDER, (y+1)*OVERRENDER)]
                    sum_ = 0
                    for subpixel in pixel:
                        sum_ += sum(subpixel)
                    pixels[-1].append(sum_/(OVERRENDER**2))
            pixels = np.array(pixels)
            #print(pixels)
        else:
            pixels = np.array(pixels_raw)

        pixels = pixels * (255 / np.max(pixels))

        im = Image.fromarray(pixels)
        im = im.convert("L")
        im.save("renders\\"+str(frame)+".png", "png")
        print("rendered frame:", frame)
