

extern crate image;
// une bibliothèque pour le traitement d'images
extern crate imageproc;


use image::{GenericImageFormat, GrayImage, ImageBuffer, Luma};
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use image::{DynamicImage, GenericImageView, Rgb};
use imageproc::definitions::Image;
use std::fs;
use std::os::raw::c_uint;
use std::time::Instant;
use image::imageops::FilterType;
use rand::Rng;
/*use crate::pmc::{Layer, Network, Neuron};*/

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
pub extern "C" fn flatten_images(image_paths: Vec<&str>) -> Vec<Vec<f32>> {
    let mut flattened_images: Vec<Vec<f32>> = Vec::new();

    for image_path in image_paths {
        let img = image::open(&Path::new(image_path)).unwrap().to_luma8();

        let mut flattened: Vec<f32> = Vec::new();

        for pixel in img.pixels() {
            flattened.push(pixel[0] as f32 / 255.0);
        }

        flattened_images.push(flattened);
    }

    flattened_images
}



#[no_mangle]
pub unsafe extern "C" fn initialize_weights (num_features: c_uint, num_classes: c_uint) -> *mut f32 {
    let num_features = num_features as usize;
    let num_classes = num_classes as usize;

    // Créez un seul vecteur pour tous les poids. Chaque ensemble consécutif de 'num_features' éléments représente les poids d'une classe.
    let mut weights: Vec<f32> = vec![0.0; num_features * num_classes];

    // Retourne un pointeur vers les poids. Les poids ne seront pas détruits lorsque cette fonction se terminera car nous avons oublié le vecteur.
    let ptr = weights.as_mut_ptr();
    std::mem::forget(weights);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn predict_image_class(image: *const f32, weights: *const f32, num_features: usize, num_classes: usize) -> usize {
    let image = std::slice::from_raw_parts(image, num_features);
    let weights = std::slice::from_raw_parts(weights, num_features * num_classes);

    // Initialize max_score to the smallest possible value for a f32
    // to ensure no calculated score is less than max_score at the beginning
    let mut max_score = std::f32::NEG_INFINITY;
    let mut predicted_class = 0;

    for class in 0..num_classes {
        let class_weights = &weights[class * num_features..(class + 1) * num_features];
        let score: f32 = image.iter().zip(class_weights.iter()).map(|(&x, &y)| x * y).sum();

        if score > max_score {
            max_score = score;
            predicted_class = class;
        }
    }

    predicted_class
}

#[no_mangle]
pub unsafe extern "C" fn train_model(
    images: *const f32,
    labels: *const c_uint,
    weights: *mut f32,
    num_samples: usize,
    num_features: usize,
    num_classes: usize,
    learning_rate: f32,
    num_iterations: usize,
) {
    let images = std::slice::from_raw_parts(images, num_samples * num_features);
    let labels = std::slice::from_raw_parts(labels, num_samples);
    let weights = std::slice::from_raw_parts_mut(weights, num_features * num_classes);

    for _ in 0..num_iterations {
        for i in 0..num_samples {
            let image = &images[i * num_features..(i + 1) * num_features];
            let true_class = labels[i] as usize;
            let predicted_class = predict_image_class(image.as_ptr(), weights.as_ptr(), num_features, num_classes);

            // If the model made a prediction error, the algorithm adjusts the weights of the predicted class and the true class
            if predicted_class != true_class {
                for j in 0..num_features {
                    weights[predicted_class * num_features + j] -= learning_rate * image[j];
                    weights[true_class * num_features + j] += learning_rate * image[j];
                }
            }
        }
    }
}


#[no_mangle]
pub unsafe extern "C" fn test_model(
    images: *const f32,
    labels: *const c_uint,
    weights: *const f32,
    num_samples: usize,
    num_features: usize,
    num_classes: usize,
) -> f32 {
    let images = std::slice::from_raw_parts(images, num_samples * num_features);
    let labels = std::slice::from_raw_parts(labels, num_samples);
    let weights = std::slice::from_raw_parts(weights, num_features * num_classes);

    let mut correct_predictions = 0;
    for i in 0..num_samples {
        let image = &images[i * num_features..(i + 1) * num_features];
        let true_class = labels[i] as usize;
        let predicted_class = predict_image_class(image.as_ptr(), weights.as_ptr(), num_features, num_classes);

        if predicted_class == true_class {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / num_samples as f32 * 100.0;
    accuracy
}


#[no_mangle]
pub extern "C" fn my_add(a: i32, b: i32) -> i32 {
    a + b
}

use rand::Rng;
use ndarray::{Array, Array2, arr2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

const EPOCHS: usize = 1000000;


pub struct Neuron {
    pub weights: Vec<f32>,
    pub delta: f32,
    pub output: f32,
    pub bias: f32,
}

impl Neuron {

    #[no_mangle]
    pub unsafe extern "C" fn new(n_inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let weights: Vec<f32> = (0..10).map(|_| rng.sample(Uniform::new(0.0, 1.0))).collect();
        let bias = rng.sample(Uniform::new(0.0, 1.0));
        Neuron {
            weights,
            delta: 0.0,
            output: 0.0,
            bias,
        }
    }


    #[no_mangle]
    pub unsafe extern "C" fn forward(&mut self, inputs: &Vec<f32>) {
        let sum: f32 = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        self.output = self.tanh(sum + self.bias);
    }


    #[no_mangle]
    pub unsafe extern "C" fn tanh(&self, x: f32) -> f32 {
        x.tanh()
    }


    #[no_mangle]
    pub unsafe extern "C" fn calculate_delta(&mut self, error: f32) {
        self.delta = (1.0 - self.output.powi(2)) * error;
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub(crate) inputs: Vec<f32>,
}

impl Layer {

    #[no_mangle]
    pub unsafe extern "C" fn new(n_inputs: usize, n_neurons: usize) -> Layer {
        let neurons: Vec<Neuron> = (0..n_neurons).map(|_| Neuron::new(n_inputs)).collect();
        Layer {
            neurons,
            inputs: vec![0.0; n_inputs],
        }
    }


    #[no_mangle]
    pub unsafe extern "C" fn forward(&mut self, inputs: &Vec<f32>) {
        self.inputs = inputs.clone();
        for neuron in self.neurons.iter_mut() {
            neuron.forward(inputs);
        }
    }


    #[no_mangle]
    pub unsafe extern "C" fn backward(&mut self, errors: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        let mut next_errors = vec![0.0; self.neurons[0].weights.len()];
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if i < 9 {
                let error = errors[i];
                for (j, weight) in neuron.weights.iter_mut().enumerate() {
                    println!("{} {}", j, weight);
                    *weight += learning_rate * error * self.inputs[j];
                    next_errors[j] += *weight * error;
                }
                neuron.bias += learning_rate * error;
            }
        }
        next_errors
    }

    #[no_mangle]
    pub unsafe extern "C" fn get_outputs(&self) -> Vec<f32> {
        self.neurons.iter().map(|neuron| neuron.output).collect()
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {

    #[no_mangle]
    pub unsafe extern "C" fn predict(&mut self, inputs: Vec<f32>) -> usize {
        let outputs = self.forward(inputs);
        outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }


    #[no_mangle]
    pub unsafe extern "C" fn accuracy(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<usize>) -> f32 {
        let mut correct_count = 0;
        for (input, &label) in inputs.iter().zip(labels) {
            let prediction = self.predict(input.clone());
            if prediction == label {
                correct_count += 1;
            }
        }
        correct_count as f32 / inputs.len() as f32
    }


    #[no_mangle]
    pub unsafe extern "C" fn new(sizes: &[usize]) -> Network {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1]));
        }
        Network { layers }
    }


    #[no_mangle]
    pub unsafe extern "C" fn forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers[0].forward(&inputs);
        for i in 1..self.layers.len() {
            let inputs = self.layers[i - 1].get_outputs();
            self.layers[i].forward(&inputs);
        }
        self.layers.last().unwrap().get_outputs()
    }
    #[no_mangle]
    pub unsafe extern "C" fn backward(&mut self, expected: &Vec<f32>, learning_rate: f32) {
        let mut errors: Vec<f32> = self
            .layers
            .last()
            .unwrap()
            .neurons
            .iter()
            .zip(expected)
            .map(|(neuron, &exp)| exp - neuron.output)
            .collect();
        for i in (0..self.layers.len()-1).rev() {
            let next_errors = self.layers[i].backward(&errors, learning_rate);
            errors = next_errors;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn train(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>, learning_rate: f32, n_epochs: usize) {
        for _ in 0..n_epochs {
            for (input, label) in inputs.iter().zip(labels) {
                self.forward(input.clone());
                self.backward(label, learning_rate);
            }
        }
    }
}

pub unsafe extern "C" fn network_predict(network: *mut Network, inputs: Vec<f32>) -> usize {

    let network = &mut network;
    network.predict(inputs)
}

#[no_mangle]
pub extern "C" fn init_network(sizes: *const usize, len: usize) -> *mut Network {
    let sizes = unsafe { slice::from_raw_parts(sizes, len) };
    let network = Network::new(sizes);
    Box::into_raw(Box::new(network));

    pub fn add(left: usize, right: usize) -> usize {
        left + right
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn it_works() {
            let result = add(2, 2);
            assert_eq!(result, 4);
        }
    }
}
