import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Wedge
from scipy.signal import butter, sosfilt, sosfilt_zi
import tkinter as tk
import sounddevice as sd
import queue

# ============================================
# Guitar strings reference
# ============================================

STRINGS = {
    "Low E": 82.41,
    "A": 110.00,
    "D": 146.83,
    "G": 196.00,
    "B": 246.94,
    "High E": 329.63
}

def cents_diff(f_meas, f_target):
    return (1200 * np.log2(f_meas / f_target)) / 10

def freq_to_string(freq):
    diffs = {note: abs(freq - f) for note, f in STRINGS.items()}
    name = min(diffs, key=diffs.get)
    return name, STRINGS[name]

# ============================================
# Bandpass filter (anti noise)
# ============================================

def make_bandpass_sos(low_hz, high_hz, fs, order=5):
    nyq = 0.5 * fs
    low = np.clip(low_hz / nyq, 0.0001, 0.99)
    high = np.clip(high_hz / nyq, 0.0002, 0.999)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sos

# ============================================
# Gauge drawing
# ============================================

def draw_gauge(fig, diff_cents, string_name):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    base = Wedge((0, 0), 1, 0, 180, color="lightgray", ec="k")
    ax.add_patch(base)

    zone_ok = Wedge((0, 0), 1, 80, 100, color="lightgreen")
    zone_low = Wedge((0, 0), 1, 100, 180, color="#ff7f7f")
    zone_hi = Wedge((0, 0), 1, 0, 80, color="#ff7f7f")
    for z in [zone_ok, zone_low, zone_hi]:
        ax.add_patch(z)

    lim = np.clip(diff_cents, -50, 50)
    theta = 90 - (lim / 50) * 90
    rad = np.deg2rad(theta)

    x = np.cos(rad)
    y = np.sin(rad)
    ax.plot([0, x], [0, y], color="black", lw=3)
    ax.plot(0, 0, "ko")

    ax.text(0, -0.15, string_name, ha="center", va="center", fontsize=20, weight="bold")
    ax.text(0, 0.9, f"{diff_cents:+.1f} cents", ha="center", fontsize=14)
    fig.tight_layout()

# ============================================
# Frequency detection using autocorrelation
# ============================================

def detect_frequency_autocorr(x, fs):
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]  # éviter le pic à zéro
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0
    return fs / peak

# ============================================
# Live tuner class
# ============================================

class LiveTuner:
    def __init__(self, fig_gauge, canvas_gauge, fig_spectrum, canvas_spectrum, selected_device_name, device_dict, fs=44100, block_duration=0.12):
        self.fs = fs
        self.block_duration = block_duration
        self.blocksize = int(self.fs * self.block_duration)
        self.q = queue.Queue()
        self.stream = None
        self.running = False
        self.fig_gauge = fig_gauge
        self.canvas_gauge = canvas_gauge
        self.fig_spectrum = fig_spectrum
        self.canvas_spectrum = canvas_spectrum
        self.selected_device_name = selected_device_name
        self.device_dict = device_dict
        self.last_note = None
        self.note_counter = 0
        self.note_confirm_threshold = 3  # nombre de blocs consécutifs pour confirmer la note


        # Passe-bande plus ciblé pour guitare
        self.sos = make_bandpass_sos(40, 400, fs)
        self.zi = sosfilt_zi(self.sos)

        # Historique pour moyenne glissante
        self.freq_history = []

    def audio_callback(self, indata, frames, time, status):
        if not status:
            self.q.put(indata.copy())

    def start(self):
        if self.running:
            return
        try:
            device_idx = self.device_dict[self.selected_device_name.get()]
            self.stream = sd.InputStream(
                device=device_idx,
                channels=1,
                samplerate=self.fs,
                blocksize=self.blocksize,
                callback=self.audio_callback
            )
            self.stream.start()
            self.running = True
        except Exception as e:
            print("Audio error:", str(e))

    def stop(self):
        if not self.running:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except:
            pass
        self.running = False
        with self.q.mutex:
            self.q.queue.clear()
        self.zi = sosfilt_zi(self.sos)
        self.freq_history = []

    def poll(self):
        last = None
        while not self.q.empty():
            last = self.q.get()

        if last is not None:
            arr = np.squeeze(last)
            peak = np.max(np.abs(arr))
            if peak < 0.02:  # seuil minimal réduit pour cordes graves
                if self.running:
                    root.after(int(self.block_duration*500), self.poll)
                return

            arr = arr / peak  # normalisation
            arr *= 2.0       # boost léger pour cordes graves

            try:
                filtered, zf = sosfilt(self.sos, arr, zi=self.zi * arr[0])
            except:
                filtered = sosfilt(self.sos, arr)
                zf = sosfilt_zi(self.sos) * filtered[-1]
            self.zi = zf

            # ----- Spectre -----
            N = len(filtered)
            fft_vals = np.abs(np.fft.rfft(filtered))
            fft_freqs = np.fft.rfftfreq(N, 1/self.fs)

            ax = self.fig_spectrum.axes[0]
            ax.clear()
            ax.plot(fft_freqs, fft_vals, color="blue")
            ax.set_xlim(0, 1500)
            ax.set_ylim(0, np.max(fft_vals)*1.1)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            self.fig_spectrum.tight_layout()
            self.canvas_spectrum.draw()

            # ----- Jauge -----
            freq = detect_frequency_autocorr(filtered, self.fs)
            if freq > 0:
                self.freq_history.append(freq)
                # on ignore les 0 dans la moyenne
                self.freq_history = [f for f in self.freq_history if f > 0]
                if len(self.freq_history) > 5:
                    self.freq_history.pop(0)
                freq_smooth = np.mean(self.freq_history)
            else:
                freq_smooth = 0

            string, target = freq_to_string(freq_smooth)

            # Vérification de stabilité
            if string == self.last_note:
                self.note_counter += 1
            else:
                self.note_counter = 1
                self.last_note = string

            if self.note_counter >= self.note_confirm_threshold:
                # On change l'affichage uniquement si la note est stable
                diff = cents_diff(freq_smooth, target) if freq_smooth > 0 else 0
                status_text.set(f"{string} : {freq_smooth:.2f} Hz ({diff:+.1f} c)")
                draw_gauge(self.fig_gauge, diff, string)
                self.canvas_gauge.draw()

        if self.running:
            root.after(int(self.block_duration*500), self.poll)

# ============================================
# Tkinter GUI
# ============================================

root = tk.Tk()
root.title("Guitar tuner + Spectrum")

# Canevas jauge
fig_gauge = plt.Figure(figsize=(6,3))
canvas_gauge = FigureCanvasTkAgg(fig_gauge, master=root)
canvas_gauge.get_tk_widget().pack()

# Canevas spectre
fig_spectrum = plt.Figure(figsize=(6,2))
ax_spec = fig_spectrum.add_subplot(111)
canvas_spectrum = FigureCanvasTkAgg(fig_spectrum, master=root)
canvas_spectrum.get_tk_widget().pack()

status_text = tk.StringVar()
status_text.set("No signal")
label_status = tk.Label(root, textvariable=status_text)
label_status.pack()

frame_btn = tk.Frame(root)
frame_btn.pack(pady=6)

# ----------------------------
# Liste des périphériques audio
# ----------------------------
input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
device_dict = {}
device_names_unique = []
for d in input_devices:
    name = d['name']
    if "primary" in name.lower() or "sound mapper" in name.lower():
        continue
    if name not in device_dict:
        device_dict[name] = d['index']
        device_names_unique.append(name)

selected_device_name = tk.StringVar()
selected_device_name.set(device_names_unique[0] if device_names_unique else "No input")

tk.Label(frame_btn, text="Select mic:").grid(row=0, column=0, sticky="w")
device_menu = tk.OptionMenu(frame_btn, selected_device_name, *device_names_unique)
device_menu.grid(row=0, column=1, padx=6)

# Live tuner
live_tuner = LiveTuner(fig_gauge, canvas_gauge, fig_spectrum, canvas_spectrum, selected_device_name, device_dict)

def toggle():
    if live_tuner.running:
        live_tuner.stop()
        btn_live.config(text="Start mic")
        status_text.set("Mic stopped")
    else:
        live_tuner.start()
        btn_live.config(text="Listening...")
        btn_live.config(text="Stop mic")
        root.after(1, live_tuner.poll)

btn_live = tk.Button(frame_btn, text="Start mic", command=toggle)
btn_live.grid(row=1, column=0, columnspan=2, pady=6)

def on_close():
    live_tuner.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

