# torn

mod contours;
mod hull;

use opencv::{imgcodecs, prelude::*, Result};

use contours::make_binary_mask;
use contours::find_biggest_contour_from_binary;
use hull::{analyze_torn_by_solidity, TornAnalysis};

fn main() -> Result<()> {
    // Read input image
    let img = imgcodecs::imread("sbi.jpeg", imgcodecs::IMREAD_COLOR)?;

    // Build binary mask
    let binary = make_binary_mask(&img)?;

    // Find biggest contour (cheque boundary)
    let biggest = find_biggest_contour_from_binary(&binary)?;

    // Analyze torn using convex hull solidity
    let analysis: TornAnalysis = analyze_torn_by_solidity(&biggest, 0.96)?;

    println!("{analysis:#?}");
    Ok(())
}

✅ File 2: src/contours.rs

use opencv::{
    core::{Mat, Point, Size, Vector},
    imgproc,
    prelude::*,
    Result,
};

use opencv::core::AlgorithmHint;

/// Converts color cheque image into a clean binary mask
/// - cheque/paper should become WHITE (255)
/// - background should become BLACK (0)
pub fn make_binary_mask(color_img: &Mat) -> Result<Mat> {
    // 1) Convert to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(
        color_img,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // 2) Otsu threshold
    let mut binary = Mat::default();
    imgproc::threshold(
        &gray,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    // 3) Morphological close (fills small gaps/holes)
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;

    imgproc::morphology_ex(
        &binary,
        &mut binary,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        2,
        opencv::core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    Ok(binary)
}

/// Finds the biggest contour from a binary image (external contours only)
pub fn find_biggest_contour_from_binary(binary: &Mat) -> Result<Vector<Point>> {
    let mut contours: Vector<Vector<Point>> = Vector::new();
    let mut hierarchy = Mat::default();

    imgproc::find_contours(
        binary,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut max_area = 0.0;
    let mut biggest: Vector<Point> = Vector::new();

    for c in contours {
        let area = imgproc::contour_area(&c, false)?;
        if area > max_area {
            max_area = area;
            biggest = c;
        }
    }

    if biggest.len() == 0 {
        return Err(opencv::Error::new(0, "No contour found"));
    }

    Ok(biggest)
}

✅ File 3: src/hull.rs
use opencv::{
    core::{Point, Vector},
    imgproc,
    Result,
};

/// Final result you want to return
#[derive(Debug, Clone)]
pub struct TornAnalysis {
    pub is_torn: bool,
    pub solidity: f64,
    pub contour_area: f64,
    pub hull_area: f64,
    pub threshold: f64,
}

/// Compute convex hull points for a contour
pub fn convex_hull_points(contour: &Vector<Point>) -> Result<Vector<Point>> {
    let mut hull = Vector::<Point>::new();

    imgproc::convex_hull(
        contour,
        &mut hull,
        false, // clockwise
        true,  // return points (NOT indices)
    )?;

    Ok(hull)
}

/// Decide torn using solidity = area(contour) / area(hull)
/// threshold is tunable (start with 0.96 for CTS-like clean images)
pub fn analyze_torn_by_solidity(contour: &Vector<Point>, threshold: f64) -> Result<TornAnalysis> {
    let contour_area = imgproc::contour_area(contour, false)?;

    // If contour too small => likely invalid/partial => treat as torn
    if contour_area < 10_000.0 {
        return Ok(TornAnalysis {
            is_torn: true,
            solidity: 0.0,
            contour_area,
            hull_area: 0.0,
            threshold,
        });
    }

    let hull = convex_hull_points(contour)?;
    let hull_area = imgproc::contour_area(&hull, false)?;

    if hull_area <= 1.0 {
        return Ok(TornAnalysis {
            is_torn: true,
            solidity: 0.0,
            contour_area,
            hull_area,
            threshold,
        });
    }

    let solidity = contour_area / hull_area;
    let is_torn = solidity < threshold;

    Ok(TornAnalysis {
        is_torn,
        solidity,
        contour_area,
        hull_area,
        threshold,
    })
}
