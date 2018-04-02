mod neuron;

struct HiddenLayer {
    pub neurons : Vec<Neuron>
}

impl HiddenLayer {
    fn add(&self, neuron : Neuron) {
        self.neurons.push(neuron);
    }
}

struct InputLayer {
    signal : f64 
}

impl InputLayer {
    fn setSignal(&mut self, signal : f64) {
        self.signal = signal; 
    }

    fn getSignal(&self) -> f64 {
        signal
    }
}
