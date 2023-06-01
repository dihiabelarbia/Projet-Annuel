use ndarray::{Array2, Array3, Axis};
use image::io::Reader as ImageReader;
use image::GrayImage;
use std::path::Path;
use ndarray_rand::{RandomExt, rand_distr::Uniform};


// Fonction pour lire les images du dataset et les convertir en Array2<f64>
fn read_images_from_dir(dir: &Path) -> Vec<Array2<f64>> {
    let mut images = Vec::new();
    for entry in dir.read_dir().expect("read_dir call failed") {
        if let Ok(entry) = entry {
            let img = ImageReader::open(entry.path())
                .unwrap()
                .decode()
                .unwrap()
                .into_luma8();
            let (width, height) = img.dimensions();
            let img = img.into_raw();
            let img = Array2::from_shape_vec((height as usize, width as usize), img).unwrap();
            let img = img.mapv(|x| x as f64 / 255.);
            images.push(img);
        }
    }
    images
}

fn prepare_data(
    triste_dir: &Path,
    heureux_dir: &Path,
    colere_dir: &Path,
) -> (Array2<f64>, Array2<f64>) {
    let triste_images = read_images_from_dir(triste_dir);
    let heureux_images = read_images_from_dir(heureux_dir);
    let colere_images = read_images_from_dir(colere_dir);

    let n_samples = triste_images.len() + heureux_images.len() + colere_images.len();
    let n_features = triste_images[0].len();

    let mut input_data = Array2::zeros((n_samples, n_features));
    let mut output_data = Array2::zeros((n_samples, 3));

    for (i, img) in triste_images.iter().enumerate() {
        input_data.row_mut(i).assign(&img.view().into_shape(n_features).unwrap());
        output_data[(i, 0)] = 1.;
    }

    for (i, img) in heureux_images.iter().enumerate() {
        input_data
            .row_mut(i + triste_images.len())
            .assign(&img.view().into_shape(n_features).unwrap());
        output_data[(i + triste_images.len(), 1)] = 1.;
    }

    for (i, img) in colere_images.iter().enumerate() {
        input_data
            .row_mut(i + triste_images.len() + heureux_images.len())
            .assign(&img.view().into_shape(n_features).unwrap());
        output_data[(i + triste_images.len() + heureux_images.len(), 2)] = 1.;
    }

    (input_data, output_data)
}


#[derive(Debug)]
struct MLP {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights1: Array2<f64>,
    weights2: Array2<f64>,
}

use rand::Rng;

fn random_array(rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        data.push(rng.gen_range(low..high));
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}


impl MLP {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weights1 = random_array(input_size, hidden_size, -1., 1.);

        let weights2 = random_array(input_size, output_size, -1., 1.);

        //  let weights2 = Array2::random((hidden_size, output_size), Uniform::new(-1., 1.));
        MLP {
            input_size,
            hidden_size,
            output_size,
            weights1,
            weights2,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1. / (1. + std::f64::consts::E.powf(-x))
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1. - x)
    }

    fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let hidden = input.dot(&self.weights1).mapv(Self::sigmoid);
        let output = hidden.dot(&self.weights2).mapv(Self::sigmoid);
        (hidden, output)
    }

    fn train(&mut self, input: &Array2<f64>, target_output: &Array2<f64>, iterations: usize) {
        for _ in 0..iterations {
            let (hidden, output) = self.forward(input);
            let output_error = target_output - &output;
            let d_output = output_error * output.mapv(Self::sigmoid_derivative);
            let hidden_error = d_output.dot(&self.weights2.t());
            let d_hidden = hidden_error * hidden.mapv(Self::sigmoid_derivative);
            self.weights1.assign(&(self.weights1.clone() + input.t().dot(&d_hidden)));
            self.weights2.assign(&(self.weights2.clone() + hidden.t().dot(&d_output)));
            /*
                        self.weights1.assign(&(self.weights1 + input.t().dot(&d_hidden)));
                        self.weights2.assign(&(self.weights2 + hidden.t().dot(&d_output)));
            */
        }
    }
}

fn main() {
    println!("ceci est le pmc");
    // Préparation des données
    let triste_dir = Path::new("C:\\Users\\Sarah\\OneDrive\\Bureau\\PAIABD\\Projet-Annuel\\dataset\\sad");
    let heureux_dir = Path::new("C:\\Users\\Sarah\\OneDrive\\Bureau\\PAIABD\\Projet-Annuel\\dataset\\happy");
    let colere_dir = Path::new("C:\\Users\\Sarah\\OneDrive\\Bureau\\PAIABD\\Projet-Annuel\\dataset\\engry");

    let (input_data, output_data) = prepare_data(triste_dir, heureux_dir, colere_dir);

    // Création et entraînement du perceptron multicouche
    let input_size = input_data.ncols();
    let hidden_size = 10;
    let output_size = 3;
    let mut mlp = MLP::new(input_size, hidden_size, output_size);
    mlp.train(&input_data, &output_data, 10000);

    // Prédiction pour une nouvelle image
    let new_image_path = Path::new("C:\\Users\\Sarah\\OneDrive\\Bureau\\PAIABD\\Projet-Annuel\\src\\sad.jpg");
    let new_image = ImageReader::open(new_image_path)
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8();
    let (width, height) = new_image.dimensions();
    let new_image = new_image.into_raw();
    let new_image =
        Array2::from_shape_vec((height as usize, width as usize), new_image).unwrap();
    let new_image = new_image.mapv(|x| x as f64 / 255.);
    let new_image = new_image.into_shape(input_size).unwrap();


    //   let new_image = new_image.into_shape((height as usize, width as usize)).unwrap();
    let new_image = new_image.into_shape((height as usize, width as usize)).unwrap();

    let (_, prediction) = mlp.forward(&new_image);
    println!("Prediction: {:?}", prediction);


}
