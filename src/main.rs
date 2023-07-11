extern crate image;
// une bibliothèque pour le traitement d'images
extern crate imageproc;
use image::Pixel;


use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use imageproc::definitions::Image;
use std::fs;
use std::time::Instant;
use image::imageops::FilterType;

// Définition de certaines constantes qui seront utilisées dans le reste du programme
const IMAGE_WIDTH: u32 = 36;
const IMAGE_HEIGHT: u32 = 36;

const NUM_CLASSES: usize = 3;
const LEARNING_RATE: f32 = 0.5;
const NUM_ITERATIONS: usize = 1;



// Fonction pour charger les images d'un dossier
fn load_images_from_folder(folder_path: &str) -> Vec<DynamicImage> {
    let mut images: Vec<DynamicImage> = Vec::new();

    for entry in fs::read_dir(folder_path).unwrap() {
        //lire chaque image du le dossier
        let entry = entry.unwrap();
        //lire le chemin exacte de chaque image
        let path = entry.path();
        let image = image::open(path).unwrap();
        //envoyer chaque image dans le vecteur
        images.push(image);
    }
    images
}
//fonction de test dll
fn my_add(a: i32, b: i32) -> i32 {
    a + b
}


// Fonction pour redimensionner toutes les images à une taille fixe
fn resize_images(images: Vec<DynamicImage>, width: u32, height: u32) -> Vec<DynamicImage> {
    let mut resized_images: Vec<DynamicImage> = Vec::new();

    for image in images {
        //Le filtre Lanczos est un filtre de redimensionnement d'image qui
        // vise à préserver le plus possible les détails de l'image d'origine lors du redimensionnemen
        let resized_image = image.resize_exact(width, height, FilterType::Lanczos3);
        // ajouter la resized_image à la fin du vecteur
        resized_images.push(resized_image);
    }
    resized_images
}

// Fonction pour convertir les images en vecteurs de pixels
// images dynamiques c'est-à-dire des images dont le type de couleur et la profondeur peuvent varier
// renvoie un vecteur de vecteurs de nombres flottants
fn flatten_images(images: Vec<DynamicImage>) -> Vec<Vec<f32>> {
    let mut flattened_images: Vec<Vec<f32>> = Vec::new();

    for image in images {
        // etre sur que l'image l'image a trois canaux de couleur
        let rgb_image = image.into_rgb8();
        let (width, height) = rgb_image.dimensions();

        let mut flattened_image: Vec<f32> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_image.get_pixel(x, y);
                // Prend la valeur du canal rouge du pixel,
                // le convertit en un nombre flottant et le normalise pour qu'il soit compris entre 0 et 1.
                // ajoute cette valeur au vecteur
                flattened_image.push(pixel[0] as f32 / 255.0);
                // idem pour canal vert
                flattened_image.push(pixel[1] as f32 / 255.0);
                // idem pour le canal bleu
                flattened_image.push(pixel[2] as f32 / 255.0);
            }
        }

        flattened_images.push(flattened_image);
    }

    flattened_images
}

fn train_linear_regression_model(
    x_ptr: *const f32,
    y_ptr: *const f32,
    num_samples: usize,
    num_features: usize,
    num_iterations: usize,
    learning_rate: f32,
) -> *mut f32 {
    // Convert raw pointers to slices
    let x = unsafe { std::slice::from_raw_parts(x_ptr, num_samples * num_features) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, num_samples) };

    // Initialize weights to zeros
    let mut weights: Vec<f32> = vec![0.0; num_features];

    // Gradient descent
    for _ in 0..num_iterations {
        let mut gradient: Vec<f32> = vec![0.0; num_features];
        for i in 0..num_samples {
            let xi = &x[i * num_features..(i + 1) * num_features];
            let yi = y[i];
            let prediction: f32 = xi.iter().zip(weights.iter()).map(|(&a, &b)| a * b).sum();
            for j in 0..num_features {
                gradient[j] += (prediction - yi) * xi[j];
            }
        }

        // Update weights
        for j in 0..num_features {
            weights[j] -= learning_rate * gradient[j] / num_samples as f32;
        }
    }

    // Return the weights as a C array
    let ptr = weights.as_mut_ptr();
    std::mem::forget(weights);
    ptr
}


// Fonction pour initialiser les poids du modèle.  Cette fonction crée une matrice de poids pour le classificateur.
// Chaque classe a un ensemble de poids associés à chaque pixel de l'image
//  initialisation à zéro
fn initialize_weights(num_features: usize, num_classes: usize) -> Vec<Vec<f32>> {
    let mut weights: Vec<Vec<f32>> = Vec::new();

    for _ in 0..num_classes {
        let mut class_weights: Vec<f32> = Vec::new();
        for _ in 0..num_features {
            class_weights.push(1.0);
        }
        // ajoute le vecteur de poids de la classe à la matrice des poids
        weights.push(class_weights);
    }
    weights
}

// Fonction pour prédire la classe d'une image, Cette fonction prend une image et les poids du classificateur, et prédit la classe de l'image.
//  la fonction utilise un modèle de classification linéaire, où la classe d'une image est déterminée
// en calculant un score pour chaque classe possible et en choisissant la classe avec le score le plus élevé
// Pour chaque classe, il calcule un score en sommant les produits des valeurs de pixel par leurs poids correspondants.
// Il prédit ensuite la classe qui a le score le plus élevé
// utilise la multiplication matricielle
fn predict_image_class(image: &[f32], weights: &[Vec<f32>]) -> usize {
    // Initialise max_score à la plus petite valeur possible pour un f32
    // pour s'assurer qu'aucun score calculé n'est inférieur à max_score au début
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
// Cette fonction utilise l'algorithme Perceptron pour ajuster les poids du modèle afin de minimiser les erreurs de prédiction
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

            // si le modèle a fait une erreur de prédiction, l'algorithme ajuste les poids de la classe prédite et de la classe réelle
            if predicted_class != true_class {
                for (pixel, weight) in image.iter().zip(weights[predicted_class].iter_mut()) {
                    *weight -= learning_rate * pixel;
                }
                // l'algorithme diminue la valeur de chaque poids proportionnellement à la valeur du pixel correspondant
                for (pixel, weight) in image.iter().zip(weights[true_class].iter_mut()) {
                    *weight += learning_rate * pixel;
                }
            }
        }
    }
}

// Fonction pour tester le modèle
fn test_model(images: &[Vec<f32>], weights: &[Vec<f32>], num_classes: usize) -> f32 {
    // initialisation du compteur pour le nombre de prédictions correctes
    let mut correct_predictions = 0;

    // Cette boucle itère sur chaque image de l'ensemble de test.
    // Elle utilise une particularité de la méthode zip où si un itérable est plus long que l'autre,
    // les éléments supplémentaires de l'itérable le plus long sont ignorés.
    // Ici, elle crée une paire d'une image et d'un nombre croissant à partir de zéro
    for (image, label) in images.iter().zip(0..) {
        let predicted_class = predict_image_class(image, weights);
        // l'algorithme vérifie si la classe prédite est égale à la véritable classe de l'image
        // qui est donnée par label % num_classes - cette opération garantit que l'étiquette est bien dans la plage de classes possibles.
        // Si la prédiction est correcte, le compteur de prédictions correctes est incrémenté de un
        if predicted_class == label % num_classes {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / images.len() as f32 * 100.0;
    accuracy
}

use rand::Rng;
use ndarray::{Array, Array2, arr2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::Path;

const EPOCHS: usize = 1000000;


pub struct Neuron {
    pub weights: Vec<f32>,
    pub delta: f32,
    pub output: f32,
    pub bias: f32,
}

impl Neuron {
    fn new(n_inputs: usize) -> Neuron {
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

    fn forward(&mut self, inputs: &Vec<f32>) {
        let sum: f32 = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        self.output = self.tanh(sum + self.bias);
    }

    fn tanh(&self, x: f32) -> f32 {
        x.tanh()
    }

    fn calculate_delta(&mut self, error: f32) {
        self.delta = (1.0 - self.output.powi(2)) * error;
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub(crate) inputs: Vec<f32>,
}

impl Layer {
    fn new(n_inputs: usize, n_neurons: usize) -> Layer {
        let neurons: Vec<Neuron> = (0..n_neurons).map(|_| Neuron::new(n_inputs)).collect();
        Layer {
            neurons,
            inputs: vec![0.0; n_inputs],
        }
    }

    fn forward(&mut self, inputs: &Vec<f32>) {
        self.inputs = inputs.clone();
        for neuron in self.neurons.iter_mut() {
            neuron.forward(inputs);
        }
    }

    fn backward(&mut self, errors: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        let mut next_errors = vec![0.0; self.neurons[0].weights.len()];
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if i < 10 {
                let error = errors[i];
                println!("{}",error);
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
    fn get_outputs(&self) -> Vec<f32> {
        self.neurons.iter().map(|neuron| neuron.output).collect()
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    fn predict(&mut self, inputs: Vec<f32>) -> usize {
        let outputs = self.forward(inputs);
        outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    fn accuracy(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<usize>) -> f32 {
        let mut correct_count = 0;
        for (input, &label) in inputs.iter().zip(labels) {
            let prediction = self.predict(input.clone());
            if prediction == label {
                correct_count += 1;
            }
        }
        correct_count as f32 / inputs.len() as f32
    }

    fn new(sizes: &[usize]) -> Network {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1]));
        }
        Network { layers }
    }

    fn forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers[0].forward(&inputs);
        for i in 1..self.layers.len() {
            let inputs = self.layers[i - 1].get_outputs();
            self.layers[i].forward(&inputs);
        }
        self.layers.last().unwrap().get_outputs()
    }

    fn backward(&mut self, expected: &Vec<f32>, learning_rate: f32) {
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

    fn train(&mut self, inputs: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>, learning_rate: f32, n_epochs: usize) {
        for _ in 0..n_epochs {
            for (input, label) in inputs.iter().zip(labels) {
                self.forward(input.clone());
                self.backward(label, learning_rate);
            }
        }
    }
}

fn load_image(file_path: &str) -> DynamicImage {
    image::open(&Path::new(file_path)).expect("Failed to load the image")
}

fn resize_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Triangle)
}

fn image_to_vec(image: DynamicImage) -> Vec<f32> {
    let rgb_image = image.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    let mut vec = Vec::with_capacity((width * height * 3) as usize);
    for pixel in rgb_image.pixels() {
        vec.push(pixel[0] as f32 / 255.0);
        vec.push(pixel[1] as f32 / 255.0);
        vec.push(pixel[2] as f32 / 255.0);
    }
    vec
}

fn main() {
    let triste_images = load_images_from_folder("C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\dataset\\sad");
    let heureux_images = load_images_from_folder("C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\dataset\\happy");
    let enerve_images = load_images_from_folder("C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\dataset\\engry");
    println!("here1");

    // Redimensionnement des images
    let triste_resized = resize_images(triste_images, IMAGE_WIDTH, IMAGE_HEIGHT);
    let heureux_resized = resize_images(heureux_images, IMAGE_WIDTH, IMAGE_HEIGHT);
    let enerve_resized = resize_images(enerve_images, IMAGE_WIDTH, IMAGE_HEIGHT);
    println!("here2");
    // Conversion des images en vecteurs de pixels
    let triste_flattened = flatten_images(triste_resized);
    let heureux_flattened = flatten_images(heureux_resized);
    let enerve_flattened = flatten_images(enerve_resized);
    println!("here3");

    //linéaire teste
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
    let mut weights = initialize_weights(IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 2, NUM_CLASSES);

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
    // Création du réseau de neurones

    //PMC teste
    let mut network = Network::new(&[IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 3, 1500, 700, 300, 3]);
    println!("here4");
    // Création des étiquettes pour les classes triste, heureuse et enervée
    let triste_labels: Vec<Vec<f32>> = (0..triste_flattened.len()).map(|_| vec![1.0, 0.0, 0.0]).collect();
    let heureux_labels: Vec<Vec<f32>> = (0..heureux_flattened.len()).map(|_| vec![0.0, 1.0, 0.0]).collect();
    let enerve_labels: Vec<Vec<f32>> = (0..enerve_flattened.len()).map(|_| vec![0.0, 0.0, 1.0]).collect();
    println!("here5");
    let inputs = [&triste_flattened[..], &heureux_flattened[..], &enerve_flattened[..]].concat();
    let labels = [&triste_labels[..], &heureux_labels[..], &enerve_labels[..]].concat();
    println!("here6");
    // Entraînement du réseau
    network.train(&inputs, &labels, LEARNING_RATE, EPOCHS);
    println!("here7");
    println!("OK");
    // Chargez votre image ici en utilisant une bibliothèque d'image appropriée
    let new_image_path = "C:\\Users\\dbelarbia\\ESGI\\pa\\Projet-Annuel\\src\\unnamed4.jpg";
    let new_image = image::open(new_image_path).expect("Impossible de charger la nouvelle image");
    let new_resized_image = new_image.resize_exact(IMAGE_WIDTH, IMAGE_HEIGHT, FilterType::Lanczos3);
    let new_flattened_image = &flatten_images(vec![new_resized_image])[0];

    // Préparez l'entrée pour la prédiction
    let inputs = new_flattened_image.clone();

    // Effectuez la prédiction
    let predicted_class = network.predict(inputs);

    // Utilisez l'indice de la classe prédite pour obtenir la classe réelle correspondante
    let classes = ["triste", "heureuse", "enervée"];
    let predicted_class_name = classes[predicted_class];

    // Affichez le résultat de la prédiction
    println!("Classe prédite : {}", predicted_class_name);
}

