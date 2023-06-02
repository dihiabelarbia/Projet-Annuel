use rand::Rng ; // on importe la librairie rand pour faire des random
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




    fn main() {

        let entree = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let mut reso = reseau::new(vec![2, 3, 1], 0.5, SIGMOIDD);

        reso.train(entree, targets, 1000);

        println!("{:?}", reso.feed_forward(vec![0.0, 0.0]));
        println!("{:?}", reso.feed_forward(vec![0.0, 1.0]));
        println!("{:?}", reso.feed_forward(vec![1.0, 0.0]));
        println!("{:?}", reso.feed_forward(vec![1.0, 1.0]));

    }
