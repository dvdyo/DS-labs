import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from scipy.signal import butter, filtfilt

def generate_harmonic(amplitude, frequency, phase, t):
    # Generates a pretty sine wave.
    return amplitude * np.sin(frequency * t + phase)

def apply_filter(signal, cutoff_freq, filter_order, sampling_freq):
    # Applies a Butterworth low-pass filter.
    nyquist = sampling_freq / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # The cutoff cannot be >= 1.0 for the filter calculation.
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    
    b, a = butter(filter_order, normalized_cutoff, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

class HarmonicVisualizer:
    def __init__(self):
        # Default parameters
        self.init_amplitude = 1.0
        self.init_frequency = 1.0
        self.init_phase = 0.0
        self.init_noise_mean = 0.0
        self.init_noise_covariance = 0.1
        self.init_cutoff_freq = 2.0
        self.init_filter_order = 4
        self.init_show_noise = True
        self.init_show_filtered = True
        
        # Live parameters
        self.amplitude = self.init_amplitude
        self.frequency = self.init_frequency
        self.phase = self.init_phase
        self.noise_mean = self.init_noise_mean
        self.noise_covariance = self.init_noise_covariance
        self.cutoff_freq = self.init_cutoff_freq
        self.filter_order = self.init_filter_order
        self.show_noise = self.init_show_noise
        self.show_filtered = self.init_show_filtered
        
        self.t = np.linspace(0, 4 * np.pi, 1000)
        self.sampling_freq = len(self.t) / (4 * np.pi) 
        self.noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_covariance), len(self.t))
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.45, right=0.9, top=0.95)
        
        y_clean_init = generate_harmonic(self.amplitude, self.frequency, self.phase, self.t)
        self.line_clean, = self.ax.plot(self.t, y_clean_init, 'b--', linewidth=2, label='Clean Harmonic')
        
        y_noisy_init = y_clean_init + self.noise
        self.line_noisy, = self.ax.plot(self.t, y_noisy_init, 'orange', linewidth=1, label='Noisy Signal')
        self.line_noisy.set_visible(self.show_noise)
        
        signal_to_filter = y_noisy_init if self.show_noise else y_clean_init
        y_filtered_init = apply_filter(signal_to_filter, self.cutoff_freq, self.filter_order, self.sampling_freq)
        self.line_filtered, = self.ax.plot(self.t, y_filtered_init, 'purple', linewidth=2, label='Filtered', alpha=0.8)
        self.line_filtered.set_visible(self.show_filtered)
        
        self.ax.set_xlim(0, 4 * np.pi)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel('Time (t)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Harmonic Function with Noise and Filtering')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.setup_widgets()
        
    def setup_widgets(self):
        slider_color = 'lightgoldenrodyellow'
        
        ax_amp = plt.axes([0.1, 0.35, 0.35, 0.03], facecolor=slider_color)
        self.slider_amp = Slider(ax_amp, 'Amplitude', 0.1, 3.0, valinit=self.init_amplitude)
        
        ax_freq = plt.axes([0.1, 0.31, 0.35, 0.03], facecolor=slider_color)
        self.slider_freq = Slider(ax_freq, 'Frequency', 0.1, 5.0, valinit=self.init_frequency)
        
        ax_phase = plt.axes([0.1, 0.27, 0.35, 0.03], facecolor=slider_color)
        self.slider_phase = Slider(ax_phase, 'Phase', 0.0, 2*np.pi, valinit=self.init_phase)
        
        ax_noise_mean = plt.axes([0.1, 0.23, 0.35, 0.03], facecolor=slider_color)
        self.slider_noise_mean = Slider(ax_noise_mean, 'Noise Mean', -0.5, 0.5, valinit=self.init_noise_mean)
        
        ax_noise_cov = plt.axes([0.1, 0.19, 0.35, 0.03], facecolor=slider_color)
        self.slider_noise_cov = Slider(ax_noise_cov, 'Noise Var', 0.01, 0.5, valinit=self.init_noise_covariance)
        
        ax_cutoff = plt.axes([0.55, 0.31, 0.35, 0.03], facecolor=slider_color)
        self.slider_cutoff = Slider(ax_cutoff, 'Filter Cutoff', 0.5, 10.0, valinit=self.init_cutoff_freq)
        
        ax_order = plt.axes([0.55, 0.27, 0.35, 0.03], facecolor=slider_color)
        self.slider_order = Slider(ax_order, 'Filter Order', 1, 8, valinit=self.init_filter_order, valfmt='%d')
        
        ax_check1 = plt.axes([0.1, 0.12, 0.15, 0.06])
        self.check_noise = CheckButtons(ax_check1, ['Show Noise'], [self.show_noise])
        
        ax_check2 = plt.axes([0.3, 0.12, 0.15, 0.06])
        self.check_filtered = CheckButtons(ax_check2, ['Show Filtered'], [self.show_filtered])
        
        ax_reset = plt.axes([0.55, 0.12, 0.1, 0.05])
        self.button_reset = Button(ax_reset, 'Reset')
        
        self.slider_amp.on_changed(self.update_plot)
        self.slider_freq.on_changed(self.update_plot)
        self.slider_phase.on_changed(self.update_plot)
        self.slider_noise_mean.on_changed(self.update_noise)
        self.slider_noise_cov.on_changed(self.update_noise)
        self.slider_cutoff.on_changed(self.update_plot)
        self.slider_order.on_changed(self.update_plot)
        self.check_noise.on_clicked(self.toggle_noise)
        self.check_filtered.on_clicked(self.toggle_filtered)
        self.button_reset.on_clicked(self.reset_parameters)

    def update_noise(self, val):
        self.noise = np.random.normal(self.slider_noise_mean.val, np.sqrt(self.slider_noise_cov.val), len(self.t))
        self.update_plot(val)

    def toggle_noise(self, label):
        self.show_noise = not self.show_noise
        self.line_noisy.set_visible(self.show_noise)
        self.update_plot(label)

    def toggle_filtered(self, label):
        self.show_filtered = not self.show_filtered
        self.line_filtered.set_visible(self.show_filtered)
        self.update_plot(label)

    def reset_parameters(self, event):
        self.slider_amp.reset()
        self.slider_freq.reset()
        self.slider_phase.reset()
        self.slider_noise_mean.reset()
        self.slider_noise_cov.reset()
        self.slider_cutoff.reset()
        self.slider_order.reset()

        if self.show_noise != self.init_show_noise:
            self.check_noise.set_active(0) 
        if self.show_filtered != self.init_show_filtered:
            self.check_filtered.set_active(0)

        self.noise = np.random.normal(self.init_noise_mean, np.sqrt(self.init_noise_covariance), len(self.t))
        self.update_plot(event)

    def update_plot(self, val):
        self.amplitude = self.slider_amp.val
        self.frequency = self.slider_freq.val
        self.phase = self.slider_phase.val
        self.cutoff_freq = self.slider_cutoff.val
        self.filter_order = int(self.slider_order.val)
        self.noise_mean = self.slider_noise_mean.val
        self.noise_covariance = self.slider_noise_cov.val

        y_clean = generate_harmonic(self.amplitude, self.frequency, self.phase, self.t)
        y_noisy = y_clean + self.noise
        
        signal_to_filter = y_noisy if self.show_noise else y_clean
        
        self.line_clean.set_ydata(y_clean)
        self.line_noisy.set_ydata(y_noisy)
        
        try:
            y_filtered = apply_filter(signal_to_filter, self.cutoff_freq, self.filter_order, self.sampling_freq)
            self.line_filtered.set_ydata(y_filtered)
        except ValueError:
            pass

        # Auto-adjust y-axis
        visible_lines = [self.line_clean]
        if self.show_noise:
            visible_lines.append(self.line_noisy)
        if self.show_filtered:
            visible_lines.append(self.line_filtered)
        
        max_y = 1.1 * max(np.max(np.abs(line.get_ydata())) for line in visible_lines)
        max_y = max(max_y, self.amplitude * 1.1) 
        
        self.ax.set_ylim(-max_y, max_y)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

if __name__ == "__main__":
    visualizer = HarmonicVisualizer()
    visualizer.show()