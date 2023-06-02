extern crate image;
// une bibliothèque pour le traitement d'images
extern crate imageproc;

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use imageproc::definitions::Image;
use std::fs;
use std::time::Instant;
use image::imageops::FilterType;

use rand::Rng;//onimportelalibrairierandpourfairedesrandom

//use serde::{Deserialize, Serialize};
//use serde_json::{from_str, json};

//use super::{activations::Activation, matrice::matrice};

// premierement , pour les test on fait une struct matrice
pub struct matrice{
    pub data: Vec<Vec<f64>>, // data est un vecteur de vecteur de f64
    pub colonne: usize,
    pub ligne: usize,
}

// specifier les caracteristique de matrice
impl matrice {
    pub fn zeros(colonne: usize, ligne: usize) -> matrice {
        matrice {
            data: vec![vec![0.0; colonne]; ligne], // vec![vec![0.0; colonne]; ligne] est un vecteur de vecteur de 0.0
            colonne,
            ligne,
        }
    }

    // une fonctions qui fait les random de matrice de zero et pour chaque element on genere des élement  entre -1 et 1
    pub fn random(colonne: usize, ligne: usize) -> matrice {
        let mut rng = rand::thread_rng(); // on fait un random
        let mut res = matrice::zeros(colonne,ligne); // on fait une matrice de 0
        for i in 0..ligne {
            for j in 0..colonne {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0; // on fait un random entre -1 et 1
            }
        }
        res
    }


    pub fn multiplication (&mut self, other: &matrice) -> matrice {

        //tester si la multiplication de deux matrices est possible
        if self.colonne != other.ligne {
            panic!("Multiplication impossible");
        }
        let mut res = matrice::zeros(self.colonne, other.ligne);

        //parcurir les deux matrices et faire la multiplication
        for i in 0..self.ligne {
            for j in 0..other.colonne {
                for k in 0..self.colonne {
                    res.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        res
    }

    pub fn add(&mut self , other: &matrice) -> matrice {
        let mut sum = 0.0;

        //tester si l'addition  de deux matrices est possible
        if self.colonne != other.colonne || self.ligne != other.ligne {
            panic!("Addition impossible");
        }
        let mut res = matrice::zeros(self.colonne, other.ligne);
        let mut sum = 0.0;
        //parcurir les deux matrices et faire la multiplication
        for i in 0..self.ligne {
            for j in 0..other.colonne {
                sum += self.data[i][j] + other.data[i][j];
                res.data[i][j] = sum;
            }
        }
        res
    }

    //on va faire le produit scalaire de deux matrices
    pub fn scalaire_multipication(&self, other: &matrice ) -> matrice {
        let mut sum = 0.0;

        if self.colonne != other.colonne || self.ligne != other.ligne {
            panic!("Addition impossible");
        }
        let mut res = matrice::zeros(self.colonne, other.ligne);

        //parcurir les deux matrices et faire la multiplication
        for i in 0..self.ligne {
            for j in 0..other.colonne {
                sum += self.data[i][j] * other.data[i][j];
                res.data[i][j] = sum;
            }
        }
        res
    }



    //on va faire la soustration de deux matrices
    pub fn soustraction(&self, other: &matrice ) -> matrice {
        let mut sum = 0.0;

        if self.colonne != other.colonne || self.ligne != other.ligne {
            panic!("Addition impossible");
        }
        let mut res = matrice::zeros(self.colonne, other.ligne);

        //parcurir les deux matrices et faire la multiplication
        for i in 0..self.ligne {
            for j in 0..other.colonne {
                sum += self.data[i][j] - other.data[i][j];
                res.data[i][j] = sum;
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> matrice {
        matrice {
            ligne: data.len(),
            colonne: data[0].len(),
            data,
        }
    }
    //on Clone la matrice et on la transforme en iterateur et pour chaque valeur on le collect
    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> matrice {
        matrice::from(
            (self.data)
                .clone()
                .into_iter()
                .map(|ligne| ligne.into_iter().map(|value| function(value)).collect())
                .collect(),
        )
    }

    //on fait la transposé de la matrice
    pub fn ma_trspose(&self) -> matrice {
        let mut res = matrice::zeros(self.ligne, self.colonne);
        for i in 0..self.ligne {
            for j in 0..self.colonne {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }


}




//on va faire la fonction d'acctivation sigmoid
 pub struct Activation<'a> {
    pub fonction: &'a dyn Fn(f64) -> f64,
    pub derivee: &'a dyn Fn(f64) -> f64,
}
pub const SIGMOIDD: Activation = Activation {
    fonction: &|x| 1.0 / (1.0 + (-x).exp()),
    derivee: &|x| x * (1.0 - x),
};

impl Clone for matrice {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            colonne: self.colonne,
            ligne: self.ligne,
        }
    }
}

//on va implémenter le reseau de neuronne
 pub struct reseau<'a> {
    poids: Vec<matrice>, // on va stocker les poids du reseau
    couche: Vec<usize>, // on va stocker les couches du reseau
    data: Vec<matrice>, // on va stocker les données du reseau
    learning_rate: f64, // on va stocker le learning rate du reseau
    biais: Vec<matrice>, // on va stocker les biais du reseau
    fct_activation : Activation<'a>
}


 impl reseau<'_> {
    pub fn new<'a>(couche: Vec<usize>, learning_rate: f64, fct_activation: Activation<'a>) -> reseau {
        let mut poids = vec![];
        let mut biais = vec![];
        for s in 0..couche.len() - 1 {
            poids.push(matrice::random(couche[s], couche[s + 1]));
            biais.push(matrice::random(1, couche[s + 1]));
        }
        reseau {
            poids,
            couche,
            data: vec![], // on va stocker les données du reseau
            biais,
            fct_activation,
            learning_rate,
        }
    }

    pub fn feed_forward(&mut self, entree: Vec<f64>) -> Vec<f64> {
        if entree.len() != self.couche[0] {
            panic!("l'entrée != couche[0]");
        }
        let mut current = matrice::from(vec![entree]).ma_trspose();
        self.data = vec![current.clone()];

        for i in 0..self.couche.len() - 1 {
            current = self.poids[i]
                .multiplication(&current)
                .add(&self.biais[i])
                .map(self.fct_activation.fonction);
            self.data.push(current.clone());
        }
        current.data[0].to_owned()  // on retourne la derniere couche c'est quoi to_owned ?
    }


    //oh tu t'es trompé de résultat ??  c une fonction de prediction
    //ceci est une fonctions qui va nous permettre de faire la propagation, et de dire si a la fin on a un bon resultat ou pas
    //et si on a des erreur de prediction

    pub fn back_propogate(&mut self, sorties: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.couche[self.couche.len() - 1] {
            panic!("taille de la target incorrecte");
        }

        let parsed = matrice::from(vec![sorties]).ma_trspose();
        let mut erreur = matrice::from(vec![targets]).ma_trspose().soustraction(&parsed);
        //on fait la derivee de la fonction d'activation
        let mut gradients = parsed.map(self.fct_activation.derivee);


        for i in (0..self.couche.len() - 1).rev() {
            gradients = gradients
                .scalaire_multipication(&erreur)
                .map(&|x| x * self.learning_rate);

            self.poids[i] = self.poids[i].add(&gradients.multiplication(&self.data[i].ma_trspose())); // the diff between what was expected and the output
            self.biais[i] = self.biais[i].add(&gradients);

            erreur = self.poids[i].ma_trspose().multiplication(&erreur);
            gradients = self.data[i].map(self.fct_activation.derivee);
        }
    }
    pub fn train(&mut self, entree: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, iterations: u16) {
        for i in 1..=iterations {
            if iterations < 100 || i % (iterations / 100) == 0 {
                println!("Epoch {} of {}", i, iterations);
            }
            for j in 0..entree.len() {
                let sortie = self.feed_forward(entree[j].clone());
                self.back_propogate(sortie, targets[j].clone());
            }
        }
    }
}




//MODELE LINEAIRES
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
