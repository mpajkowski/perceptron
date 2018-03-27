extern crate rand;

use self::rand::Rng;

pub struct Neuron {
    id      : u32,
    is_bias : bool,
    weights : Vec<f64>,
    inputs  : Vec<f64>,
    gamma   : f64,
}

impl Neuron {
    pub fn new(id: u32, input_count : usize, is_bias : bool) -> Neuron {
        let mut size = input_count;

        if is_bias {
            size += 1;
        }

        let mut rng = rand::thread_rng();
        
        let mut _weights = Vec::with_capacity(size);

        for weight in &mut _weights {
            *weight = rng.gen_range(-0.5, 0.5);
        }

        Neuron {
            id      : id,
            is_bias : is_bias,
            weights : _weights, 
            inputs  : Vec::with_capacity(size),
            gamma   : 0.0,
        }
    }

    pub fn activate(&self, x : f64) -> f64 {
        1.0 / (1.0 + x.exp())
    }

    pub fn get_summed_weigths(&self) -> f64 {
        let mut sum = 0.0;

        for (input, weight) in self.inputs.iter().zip(&self.weights) {
            sum += *input * *weight;
        }

        sum
    }

    fn derivative(&self) -> f64 {
        self.activate(self.get_summed_weigths())
            * (1.0 - self.activate(self.get_summed_weigths()))
    }

    fn update_weights(&mut self) {
        let mut rng = rand::thread_rng();

        for weight in &mut self.weights {
            *weight = rng.gen_range(-0.5, 0.5);
        }
    }
}
