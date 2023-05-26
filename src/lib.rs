extern crate image;
// une bibliothèque pour le traitement d'images
extern crate imageproc;

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use imageproc::definitions::Image;
use std::fs;
use std::time::Instant;
use image::imageops::FilterType;

// Définition de certaines constantes qui seront utilisées dans le reste du programme
const IMAGE_WIDTH: u32 = 150;
const IMAGE_HEIGHT: u32 = 150;

const NUM_CLASSES: usize = 3;
const LEARNING_RATE: f32 = 0.01;
const NUM_ITERATIONS: usize = 2;

#[no_mangle]
pub extern "C" fn resize_images(images: Vec<DynamicImage>, width: u32, height: u32) -> Vec<DynamicImage> {
    let mut resized_images: Vec<DynamicImage> = Vec::new();

    for image in images {
        let resized_image = image.resize_exact(width, height, FilterType::Lanczos3);
        resized_images.push(resized_image);
    }

    resized_images
}

#[no_mangle]
pub extern "C" fn flatten_images(images: Vec<DynamicImage>) -> Vec<Vec<f32>> {
    let mut flattened_images: Vec<Vec<f32>> = Vec::new();

    for image in images {
        let rgb_image = image.into_rgb8();
        let (width, height) = rgb_image.dimensions();

        let mut flattened_image: Vec<f32> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_image.get_pixel(x, y);
                flattened_image.push(pixel[0] as f32 / 255.0);
                flattened_image.push(pixel[1] as f32 / 255.0);
                flattened_image.push(pixel[2] as f32 / 255.0);
            }
        }

        flattened_images.push(flattened_image);
    }

    flattened_images
}

#[no_mangle]
pub extern "C" fn initialize_weights(num_features: usize, num_classes: usize) -> Vec<Vec<f32>> {
    let mut weights: Vec<Vec<f32>> = Vec::new();

    for _ in 0..num_classes {
        let mut class_weights: Vec<f32> = Vec::new();
        for _ in 0..num_features {
            class_weights.push(0.0);
        }
        weights.push(class_weights);
    }

    weights
}


#[no_mangle]
pub extern "C" fn predict_image_class(image: &[f32], weights: &[Vec<f32>]) -> usize {
    let mut max_score = std::f32::NEG_INFINITY;
    let mut predicted_class = 0;

    for (class, class_weights) in weights.iter().enumerate() {
        let mut score = 0.0;

        for (pixel, weight) in image.iter().zip(class_weights.iter()) {
            score += pixel * weight;
        }

        if score > max_score {
            max_score = score;
            predicted_class = class;
        }
    }

    predicted_class
}

#[no_mangle]
pub extern "C" fn train_model(
    images: Vec<Vec<f32>>,
    labels: Vec<usize>,
    weights: &mut [Vec<f32>],
    learning_rate: f32,
    num_iterations: usize,
) {
    let num_samples = images.len();
    let num_features = images[0].len();
    for _ in 0..num_iterations {
        for (image, label) in images.iter().zip(labels.iter()) {
            let predicted_class = predict_image_class(&image, weights);
            let true_class = *label;

            if predicted_class != true_class {
                for (pixel, weight) in image.iter().zip(weights[predicted_class].iter_mut()) {
                    *weight -= learning_rate * pixel;
                }

                for (pixel, weight) in image.iter().zip(weights[true_class].iter_mut()) {
                    *weight += learning_rate * pixel;
                }
            }
        }
    }
}


#[no_mangle]
pub extern "C" fn test_model(images: &[Vec<f32>], weights: &[Vec<f32>], num_classes: usize) -> f32 {
    let mut correct_predictions = 0;
    for (image, label) in images.iter().zip(0..) {
        let predicted_class = predict_image_class(image, weights);
        if predicted_class == label % num_classes {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / images.len() as f32 * 100.0;
    accuracy
}

#[no_mangle]
pub extern "C" fn my_add(a: i32, b: i32) -> i32 {
    a + b
}