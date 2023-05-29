mod PMC;

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

// Fonction pour charger les images d'un dossier
fn load_images_from_folder(folder_path: &str) -> Vec<DynamicImage> {
    let mut images: Vec<DynamicImage> = Vec::new();

    for entry in fs::read_dir(folder_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let image = image::open(path).unwrap();
        images.push(image);
    }
    images
}

fn my_add(a: i32, b: i32) -> i32 {
    a + b
}
// Fonction pour redimensionner toutes les images à une taille fixe
fn resize_images(images: Vec<DynamicImage>, width: u32, height: u32) -> Vec<DynamicImage> {
    let mut resized_images: Vec<DynamicImage> = Vec::new();

    for image in images {
        let resized_image = image.resize_exact(width, height, FilterType::Lanczos3);
        resized_images.push(resized_image);
    }

    resized_images
}

// Fonction pour convertir les images en vecteurs de pixels
fn flatten_images(images: Vec<DynamicImage>) -> Vec<Vec<f32>> {
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

// Fonction pour initialiser les poids du modèle.  Cette fonction crée une matrice de poids pour le classificateur.
// Chaque classe a un ensemble de poids associés à chaque pixel de l'image
fn initialize_weights(num_features: usize, num_classes: usize) -> Vec<Vec<f32>> {
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

// Fonction pour prédire la classe d'une image, Cette fonction prend une image et les poids du classificateur, et prédit la classe de l'image.
fn predict_image_class(image: &[f32], weights: &[Vec<f32>]) -> usize {
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

// Fonction pour entraîner le modèle, Cette fonction entraîne le modèle de classification.
// Elle met à jour les poids du modèle en fonction des erreurs de prédiction.
fn train_model(
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

// Fonction pour tester le modèle
fn test_model(images: &[Vec<f32>], weights: &[Vec<f32>], num_classes: usize) -> f32 {
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


// Fonction principale
fn main() {
    // Chargement des images pour chaque classe
    let triste_images = load_images_from_folder("C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\dataset\\sad");
    let heureux_images = load_images_from_folder("C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\dataset\\happy");
    let enerve_images = load_images_from_folder("C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\dataset\\engry");


    // Redimensionnement des images
    let triste_resized = resize_images(triste_images, IMAGE_WIDTH, IMAGE_HEIGHT);
    let heureux_resized = resize_images(heureux_images, IMAGE_WIDTH, IMAGE_HEIGHT);
    let enerve_resized = resize_images(enerve_images, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Conversion des images en vecteurs de pixels
    let triste_flattened = flatten_images(triste_resized);
    let heureux_flattened = flatten_images(heureux_resized);
    let enerve_flattened = flatten_images(enerve_resized);

    // Création des étiquettes pour chaque classe
    let triste_labels = vec![0; triste_flattened.len()];
    let heureux_labels = vec![1; heureux_flattened.len()];
    let enerve_labels = vec![2; enerve_flattened.len()];

    // Concaténation des images et des étiquettes
    let images: Vec<Vec<f32>> = [
        triste_flattened,
        heureux_flattened,
        enerve_flattened,
    ]
        .concat();
    let labels: Vec<usize> = [
        triste_labels,
        heureux_labels,
        enerve_labels,
    ]
        .concat();

    // Initialisation des poids
    let mut weights = initialize_weights(IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 3, NUM_CLASSES);

    // Entraînement du modèle
    let cloned_images = images.clone();
    train_model(cloned_images, labels, &mut weights, LEARNING_RATE, NUM_ITERATIONS);

    // Exemple de prédiction d'une nouvelle image
    let new_image_path = "C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\src\\sad.jpg";
    let new_image = image::open(new_image_path).expect("Impossible de charger la nouvelle image");
    let new_resized_image = new_image.resize_exact(IMAGE_WIDTH, IMAGE_HEIGHT, FilterType::Lanczos3);
    let new_flattened_image = &flatten_images(vec![new_resized_image])[0];
    let predicted_class = predict_image_class(&new_flattened_image, &weights);

    let start_time = Instant::now();

    let end_time = Instant::now();
    let execution_time = end_time.duration_since(start_time);
    let accuracy = test_model(&images, &weights, NUM_CLASSES);
    println!("Précision du modèle: {}%", accuracy);
    println!("Temps d'exécution: {:?}", execution_time);
    println!("Classe prédite: {}", predicted_class);
}
