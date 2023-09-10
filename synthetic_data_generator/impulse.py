import numpy as np
import torch
import matplotlib.pyplot as plt

class Impulse:
    def __init__(self, len: int = 100,  dt: int = 1) -> None:
        self.len = len
        self.dt = dt
        self.freq = 1 / self.dt

    def to_tensor(self, path=''):
        if not hasattr(self, 'form'):
            raise Exception('Form not set')
        
        form_tensor = torch.tensor(self.form)
        torch.save(form_tensor, path)
        
    def get_sin_form(self,start_phase: int = 0):
        self.form = np.sin(np.arange(self.len) * self.freq + start_phase)
        return self
    

    def get_unnamed_form(self):
        range_vals = np.arange(-self.len // 2, self.len // 2)
        frequency = self.freq
        denominator = range_vals + 1e-10
        numerator = np.sin(range_vals * frequency) * np.cos(2*range_vals * frequency)
        self.form = numerator / denominator
        return self

    def load_from_txt(self, path):
        self.form = np.loadtxt(path)
        return self
    
    def normalize_impulse(self):
        self.form = ((self.form - np.min(self.form)) / (np.max(self.form) - np.min(self.form)) - 0.5) * 2

    def plot_form(self):
        if not hasattr(self, 'form'):
            raise Exception('Form not set')
        
        plt.plot(self.form)
        plt.show()

def generate_impulse(impulse_type: str, length: int = 100, dt: int = 1, plot: bool = False,start_phase: int = 0) -> Impulse:
    impulse = Impulse(length, dt)
    
    if impulse_type == 'sin':
        impulse = impulse.get_sin_form(start_phase)
    elif impulse_type == 'sinc':
        impulse = impulse.get_unnamed_form()
    
    if plot:
        impulse.plot_form()
    
    return impulse


if __name__ == '__main__':
    generate_impulse('sin', 100, 100/np.pi/2)
    


    