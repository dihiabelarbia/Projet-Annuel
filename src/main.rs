<<<<<<< HEAD
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
=======
mod linear_model;

use std::fs;
use rand::seq::SliceRandom;
use std::error::Error;
use rand::distributions::Uniform;
use ndarray::{Array1, Array2, Axis, Dim, Ix, Ix1, s};
use rand::thread_rng;
use ndarray_rand::RandomExt;
use ndarray::{Array, ArrayBase, Data, Ix2, OwnedRepr};
use ndarray::prelude::*;
use rulinalg::utils;

fn load_images(folder: &str) -> Vec<Vec<Vec<u16>>> {
    // Créer un vecteur pour stocker les images
    let mut images = Vec::new();

    // Créer un vecteur intermédiaire pour stocker les images d'une même classe
    let mut class_images = Vec::new();

    // Parcourir le dossier contenant les images
    for entry in fs::read_dir(folder).unwrap() {
        let file_path = entry.unwrap().path();
        // Vérifier que le fichier est une image
        if let Some(extension) = file_path.extension() {
            if extension == "jpg" || extension == "png" || extension == "jpeg" {
                // Charger l'image et la convertir en RGB16
                let image = image::open(file_path).unwrap().into_rgb16();
                // Récupérer les données de l'image sous forme de vecteur d'entiers non signés 16 bits
                let image_data = image.into_raw();
                // Ajouter les données de l'image au vecteur intermédiaire
                class_images.push(image_data);
            }
        }
>>>>>>> a561600b0536331c83bfc6652bb6f988ec3b56a6
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

<<<<<<< HEAD
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
=======
fn train_model(train_data: &[(Array<f32, Ix1>, usize)], num_classes: usize) -> Array2<f32> {
    // Extraire les caractéristiques (entrées) et les étiquettes (cibles) à partir des données d'entraînement
    let inputs: Vec<Array<f32, Ix1>> = train_data.iter().map(|(x, _)| x.clone()).collect();
    let targets: Vec<usize> = train_data.iter().map(|(_, y)| *y).collect();

    let num_samples = inputs.len();
    let num_samples = num_samples as f32;

    let num_features = inputs[0].len();
>>>>>>> a561600b0536331c83bfc6652bb6f988ec3b56a6

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

<<<<<<< HEAD
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
=======
fn test_model(test_data: &&[(Vec<Vec<u16>>, usize)], weights: &Array<f32, Dim<[usize; 2]>>, num_classes: usize) -> f32 {
    let mut num_correct = 0;

    for (input, label) in test_data.iter() {

        let predictions = inputs .iter() .map(|x| { let flattened_x = x.into_shape((x.len(),)).unwrap(); flattened_x.dot(&weights.view()) }) .collect::<Vec<_>>();
>>>>>>> a561600b0536331c83bfc6652bb6f988ec3b56a6

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
/*
//ecrit une fonction de learning_rate
 fn learning_rate(learning_rate: f32, num_iterations: i32) -> f32 {
    let mut learning_rate = learning_rate;
    if num_iterations > 1000 {
        learning_rate = 0.001;
    }
    if num_iterations > 2000 {
        learning_rate = 0.0001;
    }
    learning_rate
}

*/


fn main() -> Result<(), Box<dyn Error>> {

<<<<<<< HEAD

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
=======
    // Les chemins des dossiers contenant les images pour chaque classe
    let happy_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\heureux";
    let sad_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\Triste";
    let engry_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\colere";

    println!("{}", "here");
    // Charger les images
    let happy_images = load_images(&happy_folder);
    let sad_images = load_images(&sad_folder);
    let engry_images = load_images(&engry_folder);

    println!("{}", "here");
    // Fusionner les images des deux classes
    let mut images = happy_images.clone();
    images.extend_from_slice(&sad_images);
    images.extend_from_slice(&engry_images);

    // Créer les étiquettes pour les images (0 pour les images heureuses et 1 pour les images tristes)
    let mut labels = vec![0; happy_images.len()];
    let mut sad_labels = vec![1; sad_images.len()];
    let mut engry_labels = vec![2; engry_images.len()];
    labels.append(&mut sad_labels);
    labels.append(&mut engry_labels);

    println!("{}", "here");
    // Mélanger les images et les étiquettes
    let mut zipped_data = images.into_iter().zip(labels.into_iter()).collect::<Vec<_>>();
    zipped_data.shuffle(&mut rand::thread_rng());

    // Diviser les données en ensemble d'entraînement et ensemble de test (80% pour l'entraînement, 20% pour le test)
    let test_size = (zipped_data.len() as f32 * 0.2) as usize;
    let train_size = (zipped_data.len() as f32 * 0.8) as usize;
    let (train_data, test_data) = zipped_data.split_at(train_size);
    println!("{}",zipped_data.len());

/*
    // Nombre de classes
    let num_classes = 3;

    // Nombre de features
let num_features = train_data[0].0.len();

    // Initialiser les poids aléatoirement
    let mut rng = rand::thread_rng();
    let mut weights = Array::random((num_features, num_classes), Uniform::new(-1.0, 1.0));

    // Nombre d'itérations d'entraînement
    let num_iterations = 1000;

    // Taux d'apprentissage
    let learning_rate = 0.01;

    // Déclaration de la variable `inputs` et extraction des caractéristiques
    let inputs: Vec<Array<f32, Ix1>> = train_data.iter().map(|(x, _)| x.clone()).collect();
    let targets: Vec<usize> = train_data.iter().map(|(_, y)| *y).collect();
    // Convertir les étiquettes en encodage one-hot
    let encoded_targets = one_hot_encode(&targets, num_classes);

    // Boucle d'entraînement


    for _ in 0..num_iterations {
        // Calculer les prédictions
        let predictions = inputs.iter().map(|x| x.dot(&weights)).collect::<Vec<_>>();

        // Calculer l'erreur

        let errors = predictions
            .iter()
            .zip(&encoded_targets)
            .map(|(predicted, target)| predicted.to_owned() - target.to_owned())
            .collect::<Vec<_>>();

        // Calculer les gradients
        let gradients = inputs
            .iter()
            .zip(&errors)
            .map(|(x, error)| x.to_owned() * error)
            .collect::<Vec<_>>();

        // Mettre à jour les poids
        for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
            *weight -= gradient.sum() * learning_rate / inputs.len() as f32;
        }


    }

    // Tester le modèle
    let accuracy = test_model(&test_data, &weights, num_classes);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    Ok(()) ;

    std::process::exit(0);
 /*
>>>>>>> a561600b0536331c83bfc6652bb6f988ec3b56a6
}



<<<<<<< HEAD
=======




fn main() -> Result<(), Box<dyn Error>> {
   //implemente un modele linéaire grace aux fonctions ci-dessus

    let happy_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\heureux";
    let sad_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\Triste";
    let engry_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\colere";

    // Charger les images

}

  */
>>>>>>> a561600b0536331c83bfc6652bb6f988ec3b56a6
