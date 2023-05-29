use image::{DynamicImage, GenericImageView};
use std::fs;

struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    output: f64,
    delta: f64,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct MultiLayerPerceptron {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl MultiLayerPerceptron {
    fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize, learning_rate: f64) -> Self {
        let mut layers = Vec::new();

        // Création de la couche d'entrée
        layers.push(Layer {
            neurons: vec![Neuron {
                weights: vec![0.0; input_size],
                bias: 0.0,
                output: 0.0,
                delta: 0.0,
            }],
        });

        // Création des couches cachées
        for &hidden_size in hidden_sizes.iter() {
            let prev_layer_size = layers.last().unwrap().neurons.len();
            let mut hidden_layer = Layer { neurons: Vec::new() };

            for _ in 0..hidden_size {
                let mut weights = Vec::new();
                for _ in 0..prev_layer_size {
                    weights.push(rand::random::<f64>() - 0.5);
                }

                hidden_layer.neurons.push(Neuron {
                    weights,
                    bias: rand::random::<f64>() - 0.5,
                    output: 0.0,
                    delta: 0.0,
                });
            }

            layers.push(hidden_layer);
        }

        // Création de la couche de sortie
        let prev_layer_size = layers.last().unwrap().neurons.len();
        let mut output_layer = Layer { neurons: Vec::new() };

        for _ in 0..output_size {
            let mut weights = Vec::new();
            for _ in 0..prev_layer_size {
                weights.push(rand::random::<f64>() - 0.5);
            }

            output_layer.neurons.push(Neuron {
                weights,
                bias: rand::random::<f64>() - 0.5,
                output: 0.0,
                delta: 0.0,
            });
        }

        layers.push(output_layer);

        MultiLayerPerceptron {
            layers,
            learning_rate,
        }
    }

    fn activate(&self, x: f64) -> f64 {
        // Fonction d'activation (sigmoid)
        1.0 / (1.0 + f64::exp(-x))
    }

    fn feed_forward(&mut self, inputs: &[f64]) {
        // Alimentation avant (feed-forward) à travers le réseau
        let mut prev_layer_outputs = inputs.to_vec();
        let mut temp_outputs = Vec::new();

        for layer in &mut self.layers {
            temp_outputs.clear();

            for neuron in &mut layer.neurons {
                let mut sum = neuron.bias;

                for (weight, prev_output) in neuron.weights.iter().zip(prev_layer_outputs.iter()) {
                    sum += weight * prev_output;
                }

                temp_outputs.push(self.activate(sum));
            }

            prev_layer_outputs = temp_outputs.clone();
            layer.neurons.iter_mut().zip(temp_outputs.iter()).for_each(|(neuron, &output)| {
                neuron.output = output;
            });
        }
    }


    fn backpropagate(&mut self, inputs: &[f64], targets: &[f64]) {
        // Rétropropagation du gradient pour ajuster les poids

        // Calcul de l'erreur de la couche de sortie
        let output_layer = self.layers.last_mut().unwrap();

        for (neuron, &target) in output_layer.neurons.iter_mut().zip(targets.iter()) {
            let output = neuron.output;
            neuron.delta = output * (1.0 - output) * (target - output);
        }

        // Calcul de l'erreur pour les couches cachées en remontant
        for layer_idx in (1..self.layers.len() - 1).rev() {
            let current_layer = &mut self.layers[layer_idx];
            let next_layer = &self.layers[layer_idx + 1];

            for (neuron_idx, neuron) in current_layer.neurons.iter_mut().enumerate() {
                let output = neuron.output;
                let mut error = 0.0;

                for next_neuron in next_layer.neurons.iter() {
                    error += next_neuron.weights[neuron_idx] * next_neuron.delta;
                }

                neuron.delta = output * (1.0 - output) * error;
            }
        }

        // Mise à jour des poids et des biais
        for layer_idx in 1..self.layers.len() {
            let current_layer = &mut self.layers[layer_idx];
            let prev_layer = &self.layers[layer_idx - 1];

            for neuron in current_layer.neurons.iter_mut() {
                for (weight, delta) in neuron.weights.iter_mut().zip(prev_layer.neurons.iter().map(|neuron| neuron.output)) {
                    *weight += self.learning_rate * delta * neuron.delta;
                }

                neuron.bias += self.learning_rate * neuron.delta;
            }
        }
    }

    fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize) {
        // Entraînement du PMC

        for epoch in 0..epochs {
            let mut error = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.feed_forward(input);
                self.backpropagate(input, target);

                for (output, &expected) in self.layers.last().unwrap().neurons.iter().map(|neuron| neuron.output).zip(target.iter()) {
                    error += (output - expected).powi(2);
                }
            }

            error /= inputs.len() as f64;

            if epoch % 100 == 0 {
                println!("Epoch: {}, Error: {}", epoch, error);
            }
        }
    }

    fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        // Prédiction à partir des entrées fournies

        self.feed_forward(input);

        self.layers.last().unwrap().neurons.iter().map(|neuron| neuron.output).collect()
    }
}

// Fonction pour charger une image et la convertir en vecteur d'attributs
fn load_image(path: &str) -> Vec<f64> {
    let img = image::open(path).expect("Failed to open image");
    let resized_img = img.resize_exact(32, 32, image::imageops::FilterType::Triangle);
    let grayscale_img = resized_img.grayscale();

    let mut attributes = Vec::new();

    for (_, _, pixel) in grayscale_img.pixels() {
        let pixel_value = pixel[0] as f64 / 255.0;
        attributes.push(pixel_value);
    }

    attributes
}

fn main() {
    // Chargement des images à partir des dossiers
    let triste_dir = "C:\\Users\\Sarah\\OneDrive\\Bureau\\PA\\dataset\\sad";
    let heureux_dir = "C:\\Users\\Sarah\\OneDrive\\Bureau\\PA\\dataset\\heureux";
    let colere_dir = "C:\\Users\\Sarah\\OneDrive\\Bureau\\PA\\dataset\\colere";

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for entry in fs::read_dir(triste_dir).expect("Failed to read image directory") {
        if let Ok(entry) = entry {
            let path = entry.path();
            let attributes = load_image(&path.to_string_lossy());
            inputs.push(attributes);
            targets.push(vec![1.0, 0.0, 0.0]); // Triste
        }
    }

    for entry in fs::read_dir(heureux_dir).expect("Failed to read image directory") {
        if let Ok(entry) = entry {
            let path = entry.path();
            let attributes = load_image(&path.to_string_lossy());
            inputs.push(attributes);
            targets.push(vec![0.0, 1.0, 0.0]); // Heureux
        }
    }

    for entry in fs::read_dir(colere_dir).expect("Failed to read image directory") {
        if let Ok(entry) = entry {
            let path = entry.path();
            let attributes = load_image(&path.to_string_lossy());
            inputs.push(attributes);
            targets.push(vec![0.0, 0.0, 1.0]); // Colère
        }
    }

    // Création du PMC avec une couche cachée de 4 neurones
    let mut mlp = MultiLayerPerceptron::new(32 * 32, vec![4], 3, 0.1);

    // Entraînement du PMC
    mlp.train(&inputs, &targets, 1000);

    // Prédiction pour une nouvelle image
    let new_image_path = "C:\\Users\\Sarah\\OneDrive\\Bureau\\PA\\src\\sad.jpg";
    let new_attributes = load_image(new_image_path);
    let prediction = mlp.predict(&new_attributes);
    println!("Prediction: {:?}", prediction);
}
