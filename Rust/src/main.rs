use std::fs::File;
use std::io::BufWriter;
use ndarray::{arr1, arr2, Array1, Array2, ArrayView1};
use obj::Obj;
//use obj::{load_obj, Obj};

const X:usize = 400;
const Y:usize = 400;

fn norm(x: ArrayView1<f64>) -> f64{
    x.dot(&x).sqrt()
}

fn normalize(x: &mut Array1<f64>){
    let norm_ = norm(x.view());
    x.mapv_inplace(|e|e/norm_);
}

fn cross_product(vec0: &Array1<f64>, vec1: &Array1<f64>) -> Array1<f64>{
    arr1(&[
        vec0[1]*vec1[2] - vec0[2]*vec1[1],
        vec0[2]*vec1[0] - vec0[0]*vec1[2],
        vec0[0]*vec1[1] - vec0[1]*vec1[0]])
}

struct Triangle{
    point0: Array1<f64>,
    point1: Array1<f64>,
    point2: Array1<f64>,
    unit_normal: Array1<f64>,
    dist_to_origin: f64,
}

impl Triangle{
    fn from_points(point0: Array1<f64>, point1: Array1<f64>, point2: Array1<f64>) -> Triangle{
        let vec0 = &point2 - &point0;
        let vec1 = &point1 - &point0;
        let mut unit_normal = cross_product(&vec0, &vec1);
        normalize(&mut unit_normal);
        let dist_to_origin = (&unit_normal*&point0).sum();
        Triangle{
            point0,
            point1,
            point2,
            unit_normal,
            dist_to_origin,
        }

    }
}

struct Ray{
    origin: Array1<f64>,
    direction: Array1<f64>,
}

impl Ray {
    fn from_origin_direction(origin: Array1<f64>, mut direction: Array1<f64>) -> Ray{
        normalize(&mut direction);
        let direction = direction;
        Ray{
            origin,
            direction
        }
    }
}

/*fn get_obj(filepath: String) -> Result<Obj, Box<dyn Error>>{
    let input_file = BufReader::new(File::open(filepath)?);
    let object = obj::load_obj(input_file)?;
    Ok(object)
}*/

fn trace_ray(t: &Vec<Triangle>, r: &Ray) -> f64{
    let mut min_dist = -1.0;
    for tri in t{
        let s = (&tri.unit_normal*&r.direction).sum();
        if s != 0.0 {  // ray intersects plane
            let dist = (&tri.dist_to_origin - (&tri.unit_normal*&r.origin).sum()) / s;
            if (min_dist == -1.0 && dist > 0.0) || (0.0 < dist && dist < min_dist){
                // is shortest found
                let intersect = &r.origin + (&r.direction*dist);
                let max_dim = arr_max(&tri.unit_normal);

                // project onto plane
                let dim0;
                let dim1;
                if max_dim == 0{
                    dim0 = 1;
                    dim1 = 2;
                } else if max_dim == 1{
                    dim0 = 0;
                    dim1 = 2;
                } else {
                    dim0 = 1;
                    dim1 = 0;
                }

                let u0:f64 = intersect[dim0] - &tri.point0[dim0];
                let u1 = &tri.point1[dim0] - &tri.point0[dim0];
                let u2 = &tri.point2[dim0] - &tri.point0[dim0];
                let v0:f64 = intersect[dim1] - &tri.point0[dim1];
                let v1 = &tri.point1[dim1] - &tri.point0[dim1];
                let v2 = &tri.point2[dim1] - &tri.point0[dim1];

                // check if in triangle
                let div = arr_det(&arr2(&[[u1, u2], [v1, v2]]));
                if div != 0.0{
                    let alpha = arr_det(&arr2(&[[u0, u2], [v0, v2]])) / div;
                    let beta = arr_det(&arr2(&[[u1, u0], [v1, v0]])) / div;
                    if alpha >= 0.0 && beta >= 0.0 && alpha + beta <= 1.0{
                        min_dist = dist;
                    }
                }


            }
        }
    }
    min_dist
}

fn arr_max(x: &Array1<f64>) -> i8{
    let mut max = 0;
    let mut max_f:f64 = 0.0;
    let mut i = 0;
    for element in x{
        if element.abs() > max_f.abs(){
            max = i;
            max_f = element.abs();

        }
        i+=1;
    }
    max
}

fn arr_det(x: &Array2<f64>) -> f64{
    x[(0,0)]*x[(1,1)] - x[(0,1)]*x[(1,0)]
}

fn triangles_from_obj(obj: Obj) -> Vec<Triangle>{
    let mut vec:Vec<Triangle> = Vec::new();
    for object in obj.data.objects{
        for group in object.groups{
            for poly in group.polys{
                vec.push(Triangle::from_points(
                    arr1(&obj.data.position[poly.0[0].0].map(|x| x as f64)[..]),
                    arr1(&obj.data.position[poly.0[1].0].map(|x| x as f64)[..]),
                    arr1(&obj.data.position[poly.0[2].0].map(|x| x as f64)[..])))
            }
        }
    }
    vec
}

fn main() {
    let obj = obj::Obj::load("data.obj").expect("error loading file");
    println!("creating triangles");
    let t = triangles_from_obj(obj);
    println!("created triangles");
    //println!("{}", obj.data.objects[0].groups[0].polys[0].0[0].0);
    //println!("{:?}", obj.data.position[obj.data.objects[0].groups[0].polys[0].0[0].0]);
    /*let t = Triangle::from_points(
        arr1(&[9.4,5.,9.3]),
        arr1(&[9.5,5.,-9.1]),
        arr1(&[-9.6,5.,9.2]));*/
    //println!("t.dist_to_origin: {}", t.dist_to_origin);
    let mut output_raw:[f64; X*Y] = [0.0; X*Y];
    let mut val;
    println!("rendering");
    for x in 0..X{
        for y in 0..Y{
            let r = Ray::from_origin_direction(
                arr1(&[2.0-(x as f64)/(X as f64/4 as f64),
                    2.0-(y as f64)/(Y as f64/4 as f64),
                    1.75]),
                arr1(&[0.0, 0.0, -1.0]));
            val = trace_ray(&t, &r);
            if val == -1.0 {
                val = 0.0;
            }
            output_raw[x*Y + y] = val;
        }

    }
    println!("rendered");
        //.to_u8().expect("conversion to u8")
    let mut max = 0.0;
    for pixel in output_raw{
        if pixel > max{
            max = pixel;
        }
    }
    let output:[u8; X*Y] = output_raw.map(|pixel| (pixel*(255.0/max as f64)) as u8);
    let file = BufWriter::new(File::create("test.png").unwrap());
    let encoder = image::codecs::png::PngEncoder::new(file);
    encoder.encode(&output, X as u32, Y as u32, image::ColorType::L8)
        .expect("error encoding");
}