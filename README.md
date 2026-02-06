use opencv::{
    core::{Mat, Point, Rect},
    imgproc,
    prelude::*,
    Result,
};
use opencv::core::AlgorithmHint;
use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::gapi::Scalar;
use opencv::core::VecN;
use opencv::core::Size;
fn main() -> Result<()> {
    // 1️⃣ Read image in color (so we can draw red)
    let mut img = imgcodecs::imread("sbi.jpeg", imgcodecs::IMREAD_COLOR)?;

    // 2️⃣ Convert to grayscale for contour detection
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0,AlgorithmHint::ALGO_HINT_DEFAULT)?;
        // 3️⃣ Threshold using Otsu
    let mut binary = Mat::default();
    imgproc::threshold(&gray, &mut binary, 0.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;

    // 4️⃣ Morphological closing to fill gaps
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(5, 5), Point::new(-1, -1))?;
    imgproc::morphology_ex(&binary.clone(), &mut binary, imgproc::MORPH_CLOSE, &kernel, Point::new(-1, -1), 2, opencv::core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
    // 2️⃣ Find biggest contour
    let biggest_contour = find_biggest_contour(&gray)?;

    // 3️⃣ Draw the contour on the color image
    let mut contours_vec = Vector::<Vector<Point>>::new();
    let mut contour_points = Vector::<Point>::new();
    for p in &biggest_contour {
        contour_points.push(p);
    }
    contours_vec.push(contour_points);



        // Decide torn using convex hull
    let (is_torn, solidity, area_c, area_h) = is_torn_by_convex_hull(&biggest_contour)?;

    println!(
        "contour_area={:.0}, hull_area={:.0}, solidity={:.4}, torn={}",
        area_c, area_h, solidity, is_torn
    );
    //draw_contour(&mut img, &contour, Scalar::new(0.0, 0.0, 255.0, 0.0), 2)?;
    let hull = convex_hull_points(&biggest_contour)?;
    //draw_contour(&mut img, &hull, Scalar::new(0.0, 255.0, 0.0, 0.0), 2)?;

    let red = VecN::<f64, 4>::from([0.0, 0.0, 255.0, 0.0]); // BGR + alpha
    imgproc::draw_contours(
        &mut img,
        &contours_vec,
        0,         // index of contour
        red,       // color as VecN<f64,4>
        2,         // thickness
        imgproc::LINE_8,
        &Mat::default(),
        i32::MAX,
        Point::new(0, 0),
    )?;

    // 4️⃣ Save the output image
    imgcodecs::imwrite("biggest_contour.jpg", &img, &Vector::new())?;
    println!("Biggest contour drawn and saved as biggest_contour.jpg");

    Ok(())
}

// fn main() -> Result<()> {
//     let img = opencv::imgcodecs::imread("sbi.jpeg", opencv::imgcodecs::IMREAD_GRAYSCALE)?;
//     let biggest_contour = find_biggest_contour(&img)?;

//     println!("Biggest contour has {} points", biggest_contour.len());
//     for point in &biggest_contour {
//         println!("Point: ({}, {})", point.x, point.y);
//     }

//     Ok(())
// }

fn find_biggest_contour(image: &Mat) -> Result<Vector<Point>> {
    // Threshold
    let mut binary = Mat::default();
    imgproc::threshold(image, &mut binary, 127.0, 255.0, imgproc::THRESH_BINARY)?;

    // 1️⃣ Create Vector<Vector<Point>> for contours
    let mut contours: Vector<Vector<Point>> = Vector::new();
    let mut hierarchy = Mat::default();

    imgproc::find_contours(
        &binary,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    // 2️⃣ Find the biggest contour
    let mut biggest_contour: Vec<Point> = Vec::new();
    let mut max_area = 0.0;

    for contour in contours {
        let area = imgproc::contour_area(&contour, false)?;
        if area > max_area {
            max_area = area;
            biggest_contour = contour.to_vec();
        }
    }

    Ok(biggest_contour.into())
}
fn contour_area_rust(   contour: &[Point]) -> Result<f64> {
    // Convert to Mat
    let mat = points_to_mat(contour);
    Ok(imgproc::contour_area(&mat, false)?)
}
// Convert Vec<Point> -> Mat

fn points_to_mat(points: &[Point]) -> Mat {
    // Create Vector<Point>
    let mut v = Vector::<Point>::new();
    for &p in points {
        v.push(p);
    }

    // Convert Vector<Point> -> slice -> Mat
    let slice: &[Point] = v.as_slice();
    let mat = Mat::from_slice(slice).unwrap();
   let a= mat.try_clone().unwrap();

   a
}

fn convex_hull_points(contour: &Vector<Point>) -> Result<Vector<Point>> {
    let mut hull = Vector::<Point>::new();

    // return_points=true => hull is points
    imgproc::convex_hull(
        contour,
        &mut hull,
        false, // clockwise
        true,  // return points (NOT indices)
    )?;
    Ok(hull)
}

/// Torn decision using solidity = area(contour)/area(hull)
/// You MUST tune threshold for your scanner.
fn is_torn_by_convex_hull(contour: &Vector<Point>) -> Result<(bool, f64, f64, f64)> {
    let area_c = imgproc::contour_area(contour, false)?;

    // if too small, reject
    if area_c < 10_000.0 {
        return Ok((true, 0.0, area_c, 0.0));
    }

    let hull = convex_hull_points(contour)?;
    let area_h = imgproc::contour_area(&hull, false)?;

    if area_h <= 1.0 {
        return Ok((true, 0.0, area_c, area_h));
    }

    let solidity = area_c / area_h;

    // Typical starting thresholds:
    // - clean cheque: 0.97 - 1.00
    // - torn corner:  0.85 - 0.95 (depends how big tear is)
    //
    // Start with 0.96 and adjust with your dataset.
    let torn = solidity < 0.96;

    Ok((torn, solidity, area_c, area_h))
}

fn draw_contour(img: &mut Mat, contour: &Vector<Point>, color: VecN<f64, 4>, thickness: i32) -> Result<()> {
    let mut contours_vec = Vector::<Vector<Point>>::new();
    contours_vec.push(contour.clone());

    // draw_contours in your version can accept Scalar too; if not, switch to VecN like you did.
    imgproc::draw_contours(
        img,
        &contours_vec,
        0,
        color,
        thickness,
        imgproc::LINE_8,
        &Mat::default(),
        i32::MAX,
        Point::new(0, 0),
    )?;
    Ok(())
}
