# neuromorf/ir.py

class Neuron:
    def __init__(self, id, neuron_type, threshold):
        self.id = id
        self.type = neuron_type
        self.threshold = threshold
    
    def __repr__(self):
        return f"Neuron(id={self.id}, type={self.type}, threshold={self.threshold})"


class Synapse:
    def __init__(self, src, dst, weight):
        self.src = src
        self.dst = dst
        self.weight = weight
    
    def __repr__(self):
        return f"Synapse(src={self.src}, dst={self.dst}, weight={self.weight})"


class NeuromorphIR:
    def __init__(self):
        self.neurons = []
        self.synapses = []
    
    def add_neuron(self, neuron):
        self.neurons.append(neuron)
    
    def add_synapse(self, synapse):
        self.synapses.append(synapse)
    
    def __repr__(self):
        return f"NeuromorphIR(neurons={len(self.neurons)}, synapses={len(self.synapses)})"