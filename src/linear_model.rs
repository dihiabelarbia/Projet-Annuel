/*use std::error::Error;
use crate::{test_model, train_model};

fn main() -> Result<(), Box<dyn Error>> {
    // Charger les données
    let (train_data, test_data) = load_data()?;

    // Définir les hyperparamètres
    let num_iterations = 100;
    let learning_rate = 0.01;

    // Entraîner le modèle
    let weights = train_model(&train_data, num_iterations, learning_rate);

    // Tester le modèle
    let accuracy = test_model(&test_data, &weights);

    println!("Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

 */