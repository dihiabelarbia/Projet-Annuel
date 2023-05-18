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
    }

    // Ajouter le vecteur intermédiaire au vecteur principal
    images.push(class_images);

    // Retourner le vecteur principal
    images
}

fn flatten_images(images: Vec<Array<f32, ndarray::Dim<[usize; 3]>>>) -> ArrayBase<OwnedRepr<f32>, Ix2> {
    let num_images = images.len();
    let image_shape = images[0].shape();
    let num_features = image_shape[0] * image_shape[1] * image_shape[2];

    let mut flattened_images = Array::zeros((num_images, num_features));
    for (i, image) in images.into_iter().enumerate() {
        let flattened_image = image
            .into_shape((num_features,))
            .expect("Failed to reshape image");
        flattened_images.row_mut(i).assign(&flattened_image);
    }
    flattened_images
}


fn one_hot_encode(labels: &[usize], num_classes: usize) -> Array<f32, ndarray::Dim<[usize; 2]>> {
    let num_labels = labels.len();
    let mut encoded_labels = Array::zeros((num_labels, num_classes));

    for (i, &label) in labels.iter().enumerate() {
        encoded_labels[[i, label]] = 1.0;
    }

    encoded_labels
}

fn train_model(train_data: &[(Array<f32, Ix1>, usize)], num_classes: usize) -> Array2<f32> {
    // Extraire les caractéristiques (entrées) et les étiquettes (cibles) à partir des données d'entraînement
    let inputs: Vec<Array<f32, Ix1>> = train_data.iter().map(|(x, _)| x.clone()).collect();
    let targets: Vec<usize> = train_data.iter().map(|(_, y)| *y).collect();

    let num_samples = inputs.len();
    let num_samples = num_samples as f32;

    let num_features = inputs[0].len();

    // Convertir les étiquettes en encodage one-hot
    let encoded_targets = one_hot_encode(&targets, num_classes);

    // Initialiser les poids aléatoirement
    let mut rng = rand::thread_rng();
    let mut weights = Array::random((num_features, num_classes), Uniform::new(-1.0, 1.0));
    // Nombre d'itérations d'entraînement
    let num_iterations = 1000;

    // Taux d'apprentissage
    let learning_rate = 0.01;

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
            *weight -= gradient.sum() * learning_rate / num_samples as f32;
        }
    }
    weights
}

fn test_model(test_data: &&[(Vec<Vec<u16>>, usize)], weights: &Array<f32, Dim<[usize; 2]>>, num_classes: usize) -> f32 {
    let mut num_correct = 0;

    for (input, label) in test_data.iter() {

        let predictions = inputs .iter() .map(|x| { let flattened_x = x.into_shape((x.len(),)).unwrap(); flattened_x.dot(&weights.view()) }) .collect::<Vec<_>>();

        // Recherche de l'indice de la classe prédite avec la plus haute valeur
        let mut predicted_class = 0;
        let mut max_value = prediction[0];
        for i in 1..num_classes {
            if prediction[i] > max_value {
                max_value = prediction[i];
                predicted_class = i;
            }
        }

        if predicted_class == *label {
            num_correct += 1;
        }
    }

    let accuracy = num_correct as f32 / test_data.len() as f32;
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
}







fn main() -> Result<(), Box<dyn Error>> {
   //implemente un modele linéaire grace aux fonctions ci-dessus

    let happy_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\heureux";
    let sad_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\Triste";
    let engry_folder = "C:\\Users\\Sarah\\OneDrive\\Bureau\\Projet-Annuel-master_officiel\\dataset\\colere";

    // Charger les images

}

  */