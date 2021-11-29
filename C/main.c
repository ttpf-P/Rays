#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <png.h>

double max = 0;

const int X = 400;
const int Y = 400;
const int FRAMES = 1;
int triangle_number = 0;

const int bit_depth = 8;

const double FALLOFF_MULT = 10.;


struct ray{
    double origin[3];
    double unit_direction[3];
    double magnitude;
};

struct ray create_ray(const double origin[3], const double direction[3]){
    //init
    struct ray r;
    double magnitude;

    //origin
    memcpy(r.origin, origin, 3*sizeof(double));

    //unit direction
    magnitude = sqrt(pow(direction[0],2)+ pow(direction[1],2)+ pow(direction[2],2));
    r.unit_direction[0] = direction[0]/magnitude;
    r.unit_direction[1] = direction[1]/magnitude;
    r.unit_direction[2] = direction[2]/magnitude;
    r.magnitude = magnitude;
    return r;
}


struct triangle {
    double point0[3];
    double point1[3];
    double point2[3];

    double unit_normal[3];
    double dist_to_origin;
};

double* cross_product(const double vector0[3], const double vector1[3]){
    double *product = malloc(3*sizeof(double));
    if (product == NULL){
        printf("null");
    }
    product[0] = vector0[1]*vector1[2] - vector0[2]*vector1[1];
    product[1] = vector0[2]*vector1[0] - vector0[0]*vector1[2];
    product[2] = vector0[0]*vector1[1] - vector0[1]*vector1[0];
    return product;
}

struct triangle triangle_from_points(const double point0[3], const double point1[3], const double point2[3]) {
    //init
    struct triangle t;
    double magnitude;
    double *product;
    double vector1[3] = {point1[0]-point0[0], point1[1]-point0[1], point1[2]-point0[2]};
    double vector0[3] = {point2[0]-point0[0], point2[1]-point0[1], point2[2]-point0[2]};

    //points
    memcpy(t.point0, point0, 3*sizeof(double));
    memcpy(t.point1, point1, 3*sizeof(double));
    memcpy(t.point2, point2, 3*sizeof(double));

    //unit normal
    product = cross_product(vector1, vector0);
    magnitude = sqrt(pow(product[0],2)+ pow(product[1],2)+ pow(product[2],2));
    t.unit_normal[0] = product[0]/magnitude;
    t.unit_normal[1] = product[1]/magnitude;
    t.unit_normal[2] = product[2]/magnitude;

    //distance to origin
    t.dist_to_origin = t.unit_normal[0]*point0[0]+t.unit_normal[1]*point0[1]+t.unit_normal[2]*point0[2];

    //keep track of number of triangles
    triangle_number++;
    return t;
}


double trace_ray(struct triangle triangles[], struct ray r){
    double s;
    double dist;
    double min_dist = 9007199254740993.0;
    double intersect[3];
    double u[3], v[3];
    double div;
    double alpha, beta;
    for (int i = 0; i < triangle_number; i++) {
        s = triangles[i].unit_normal[0]*r.unit_direction[0] + triangles[i].unit_normal[1]*r.unit_direction[1] +
            triangles[i].unit_normal[2]*r.unit_direction[2];
        if (s){ //not parallel
            dist = (triangles[i].dist_to_origin -
                    (triangles[i].unit_normal[0]*r.origin[0] +
                     triangles[i].unit_normal[1]*r.origin[1] +
                     triangles[i].unit_normal[2]*r.origin[2])) / s;
            /*if (dist < 0){
                dist = -dist;
            }*/
            dist = dist;
            if ((min_dist > dist && dist > 0)){
                intersect[0] = r.origin[0] + (r.unit_direction[0] * dist);
                intersect[1] = r.origin[1] + (r.unit_direction[1] * dist);
                intersect[2] = r.origin[2] + (r.unit_direction[2] * dist);

                //map to two dimensions    use absolute value for check
                if (fabs(triangles[i].unit_normal[0]) >= fabs(triangles[i].unit_normal[1]) &&
                    fabs(triangles[i].unit_normal[0]) >= fabs(triangles[i].unit_normal[2])){
                    u[0] = intersect[1] - triangles[i].point0[1];
                    u[1] = triangles[i].point1[1] - triangles[i].point0[1];
                    u[2] = triangles[i].point2[1] - triangles[i].point0[1];

                    v[0] = intersect[2] - triangles[i].point0[2];
                    v[1] = triangles[i].point1[2] - triangles[i].point0[2];
                    v[2] = triangles[i].point2[2] - triangles[i].point0[2];
                } else if (fabs(triangles[i].unit_normal[1]) >= fabs(triangles[i].unit_normal[0]) &&
                           fabs(triangles[i].unit_normal[1]) >= fabs(triangles[i].unit_normal[2])) {
                    u[0] = intersect[0] - triangles[i].point0[0];
                    u[1] = triangles[i].point1[0] - triangles[i].point0[0];
                    u[2] = triangles[i].point2[0] - triangles[i].point0[0];

                    v[0] = intersect[2] - triangles[i].point0[2];
                    v[1] = triangles[i].point1[2] - triangles[i].point0[2];
                    v[2] = triangles[i].point2[2] - triangles[i].point0[2];
                } else {
                    u[0] = intersect[1] - triangles[i].point0[1];
                    u[1] = triangles[i].point1[1] - triangles[i].point0[1];
                    u[2] = triangles[i].point2[1] - triangles[i].point0[1];

                    v[0] = intersect[0] - triangles[i].point0[0];
                    v[1] = triangles[i].point1[0] - triangles[i].point0[0];
                    v[2] = triangles[i].point2[0] - triangles[i].point0[0];
                }
                div = (u[1]*v[2]) - (u[2]*v[1]);
                if (div){
                    alpha = ((u[0]*v[2]) - (u[2]*v[0])) / div;
                    beta = ((u[1]*v[0]) - (u[0]*v[1])) / div;
                    if (alpha >= 0 && beta >= 0 && (alpha + beta) <= 1){
                        min_dist = dist;
                    }
                }
            }
        }
    }
    if (min_dist == 9007199254740993.0){
        min_dist = 0;
    }
    //printf("%f", min_dist);
    return min_dist;
}

struct triangle* get_triangles(){
    int triangle_num;
    FILE *fp;
    fp = fopen("triangles.num", "r");
    fscanf(fp,"%d", &triangle_num);
    fclose(fp);
    printf("\n%d", triangle_num);

    double* data = malloc(sizeof(double)*triangle_num);
    fp = fopen("triangles.data", "rb");
    fread(data, sizeof(double), triangle_num, fp);
    fclose(fp);
    printf("\n%f", data[0]);
    fflush(stdout);
    struct triangle* triangles = malloc(sizeof(struct triangle)*(triangle_num/3));
    double point0[3], point1[3], point2[3];
    for (int i = 0; i < triangle_num/9; ++i) { // for every whole triangle
        point0[0] = data[i*9];
        point0[1] = data[i*9+1];
        point0[2] = data[i*9+2];
        point1[0] = data[i*9+3];
        point1[1] = data[i*9+4];
        point1[2] = data[i*9+5];
        point2[0] = data[i*9+6];
        point2[1] = data[i*9+7];
        point2[2] = data[i*9+8];
        triangles[i] = triangle_from_points(point0, point1, point2);
    }
    free(data);
    return triangles;

}

int main() {
    //ray origin
    double origin[3] = {0,0,1.75}, direction[3] = {1,1,-1};

    //lights
    double intersect[3];
    double lights[] = {-5,-5,-5};
    int light_num = 1;
    double light_vec[3];
    double brightness;
    double dist;

    //other stuff
    struct ray r;
    struct triangle* t_;
    double pre_light_dist;
    t_ = get_triangles();


    FILE *fp;
    char filename[] = "render2/pngder00000000.png";

    /*t[0] = triangle_from_points(point0, point1, point2);
    t[1] = triangle_from_points(point3, point4, point2);*/

    double (*output) = malloc(sizeof(double) * X * Y);
    int (*outputInts) = malloc(sizeof(int) * X * Y);

    for (int frame = 0; frame < FRAMES; ++frame) {
        max = 0;
        origin[0] = 2*((double )frame/FRAMES)-1;

        for (int x = 0; x < X; ++x) {
            direction[1] = -(-1 + (2.0 * x / X));
            for (int y = 0; y < Y; ++y) {
                direction[0] = -1 + (2.0 * y / Y) - (1 * (frame / (double )FRAMES)- 0.5);
                r = create_ray(origin, direction);
                pre_light_dist = trace_ray(t_, r);  // find dist

                // calculate light
                if (pre_light_dist){
                    intersect[0] = origin[0] + direction[0]*pre_light_dist;
                    intersect[1] = origin[1] + direction[1]*pre_light_dist;
                    intersect[2] = origin[2] + direction[2]*pre_light_dist;
                    brightness = 0.;
                    for (int i = 0; i < light_num; ++i) {
                        light_vec[0] = lights[i * 3] - intersect[0];
                        light_vec[1] = lights[i * 3 + 1] - intersect[0];
                        light_vec[2] = lights[i * 3 + 2] - intersect[0];

                        r = create_ray(intersect, light_vec);
                        dist = trace_ray(t_, r);

                        //printf("%f\n", sqrt(pow(light_vec[0], 2) + pow(light_vec[1], 2) + pow(light_vec[2], 2)));
                        if (dist >= r.magnitude || dist == 0) {
                            //brightness += r.magnitude * FALLOFF_MULT;
                            brightness += (5*FALLOFF_MULT)-(sqrt(r.magnitude) * FALLOFF_MULT);
                        }
                        if (brightness > max){
                            max = brightness;
                            //printf("found");
                        }
                    }
                    if (brightness == 0){
                        brightness = 1;
                    }


                } else {
                    brightness = 0;
                }

                //output[x * Y + y] = trace_ray(t_, r);
                output[x * Y + y] = brightness;
            }
            if (x % 100 == 0 && X > 1000) {
                printf("\nline: %d", x);
            }

        }
        sprintf(&filename[14], "%08d.png", frame);
        fp = fopen(filename, "wb+");

        printf("\nrendered frame: %d", frame);


        /*for (int i = 0; i < X*Y; ++i) {
            if (output[i] > max) {
                max = output[i];
                printf("found");
            }
        }*/
        //printf("%f", max);

        if (max != 0.){
            for (int i = 0; i < X*Y; ++i) {
                output[i] = output[i]*(255/max);
            }
        }

        for (int i = 0; i < X*Y; ++i) {
            outputInts[i] = (int) floor(output[i]);
        }
        //libpng stuff
        png_structp png_ptr = png_create_write_struct(
                PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!png_ptr || !info_ptr){
            return 2;
        }
        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, Y, X, bit_depth,
                     PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_bytepp row_pointers = (png_bytepp)png_malloc(png_ptr, sizeof(png_bytepp) * X);
        fflush(stdout);
        for (int i = 0; i < X; i++) {
            row_pointers[i] = (png_bytep)png_malloc(png_ptr, Y);
        }
        fflush(stdout);
        for (int hi = 0; hi < Y; hi++) {
            for (int wi = 0; wi < X; wi++) {
                //printf("\n %d \t %d", hi, wi);
                //fflush(stdout);
                row_pointers[wi][hi] = outputInts[hi + Y * wi];
            }
        }
        fflush(stdout);
        png_write_info(png_ptr, info_ptr);
        png_write_image(png_ptr, row_pointers);
        png_write_end(png_ptr, info_ptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
    }

    free(outputInts);
    free(output);
    fp = fopen("renders/renders.res", "wb+");
    fprintf(fp, "%d;%d", X, Y);
    fclose(fp);

    fp = fopen("renders/renders.frames", "wb+");
    fprintf(fp, "%d", FRAMES);
    fclose(fp);

    printf("\n");
    fflush(stdout);

    return 0;
}
