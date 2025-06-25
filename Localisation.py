import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from itertools import combinations

# Function to calculate GCC-PHAT
def gcc_phat(x, y, fs=1, max_tau=None):
    N = len(x) + len(y) - 1
    X = np.fft.fft(x, N)
    Y = np.fft.fft(y, N)
    
    # Compute the cross-power spectrum
    R = X * np.conj(Y)
    
    # Apply phase transform (PHAT)
    R = R / (np.abs(R) + 1e-8)  # Avoid division by zero
    
    # Inverse FFT to get the cross-correlation
    r = np.fft.ifft(R)
    r = np.fft.fftshift(r)  # Center the zero delay
    r = np.real(r)  # Take the real part
    
    # Find the time delay index
    max_shift = int(N / 2)
    if max_tau:
        max_shift = min(max_shift, int(fs * max_tau))
    delay_idx = np.argmax(r[max_shift:-max_shift]) - max_shift

    delay = delay_idx / fs  # Convert to seconds
    return delay, r

# Triangulation to compute azimuth/elevation from TDOA
def triangulate(tdoas, mic_positions, c=343):
    """
    Use the Time Difference of Arrival (TDOA) to triangulate the source position.
    
    :param tdoas: List of Time differences of arrival between microphone pairs
    :param mic_positions: 3D positions of microphones (6 in total)
    :param c: Speed of sound (343 m/s at 20°C)
    :return: Estimated azimuth and elevation (in degrees)
    """
    # Placeholder for triangulation logic
    # Replace this with an optimization or least-squares approach to estimate the source location
    azimuth = 45  # Example azimuth in degrees
    elevation = 30  # Example elevation in degrees
    
    return azimuth, elevation

# Example usage
if __name__ == "__main__":
    # Example signals (replace with actual signals)
    fs = 44100  # Example sample rate (Hz)
    mic_signals = [np.random.randn(10000) for _ in range(6)]  # Simulated microphone data
    
    # Microphone positions in 3D space
    mic_positions = np.array([
        [6.12, -3.54, 7.07],   # Mic 1
        [-6.12, 3.54, -7.07],  # Mic 2
        [0, 7.07, 7.07],       # Mic 3
        [0, -7.07, -7.07],     # Mic 4
        [-6.12, -3.54, 7.07],  # Mic 5
        [6.12, 3.54, 7.07]     # Mic 6
    ])
    
    # Compute TDOA for each pair of microphones
    tdoas = []
    for (i, j) in combinations(range(len(mic_positions)), 2):
        # Compute the delay using GCC-PHAT
        delay, cross_corr = gcc_phat(mic_signals[i], mic_signals[j], fs)
        tdoas.append(delay)  # Delay in seconds

        # Visualize the cross-correlation for each pair
        plt.plot(cross_corr, label=f"Mic Pair {i+1}-{j+1}")
        print(f"TDOA for Mic Pair {i+1}-{j+1}: {delay:.6f} s")

    plt.title("Cross-Correlation with Phase Transform (GCC-PHAT)")
    plt.xlabel("Sample Lag")
    plt.ylabel("Cross-correlation")
    plt.legend()
    plt.show()
    
    # Compute distance differences based on TDOA
    c = 343  # Speed of sound in m/s
    distance_differences = [tdoa * c for tdoa in tdoas]

    print("Distance Differences (m):", distance_differences)

    # Triangulate the sound source position
    azimuth, elevation = triangulate(tdoas, mic_positions)
    
    print(f"Estimated Azimuth: {azimuth}°")
    print(f"Estimated Elevation: {elevation}°")