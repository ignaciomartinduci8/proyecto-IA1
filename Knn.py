import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice
from sklearn.decomposition import PCA

class Knn:

    def __init__(self):

        self.y = None
        self.sr = None

        self.zcr = None
        self.duration = None
        self.spectralCentroid = None
        self.promCentroids1 = None

        pass

    def extractAudioData(self, file_path, graph=False, save=False):

        # Carga y preprocesamiento

        filename = file_path.split("/")[-1].replace(".wav","")

        self.y, self.sr = librosa.load(file_path, sr=44100)

        self.preprocessAudio(filename=filename, graph=graph, save=save)

        if len(self.y) > 50000:

            sounddevice.play(0.5 * np.sin(2 * np.pi * 1000 * np.arange(5000) / 44100), samplerate=44100, blocking=True)
            sounddevice.wait()
            raise Exception(f"Archivo  {filename } demasiado largo. Verificar posibles sonidos exteriores.")

        # Extracción de características

        self.zeroCrossingRate(graph=graph, filename=filename, save=save)

        self.calcDuration(graph=graph, filename=filename, save=save)

        self.temporalEvolve(graph=graph, filename=filename, save=save)

        self.spectralEvolve(graph=graph, filename=filename, save=save)

    # Métodos de extracción de características

    def preprocessAudio(self, graph=False, filename=None, save=False):

        plt.plot(self.y, alpha=0.3)

        self.y, _ = librosa.effects.trim(self.y, top_db=17)
        self.y = np.array(self.y)
        #self.y = librosa.effects.preemphasis(self.y)
        self.y = librosa.util.normalize(self.y)

        if graph or save:

            plt.plot(self.y, alpha=0.7)
            plt.xlabel("Tiempo (muestras)")
            plt.ylabel("Amplitud")
            plt.title(f"Señal de audio - {filename}")
            plt.legend(["Original", "Preprocesada"])

        if graph:
            plt.show()
            plt.close()

        if save:
            plt.savefig(f"./props/preProcessing/{filename}.png")
            plt.close()

    def zeroCrossingRate(self, graph=False, filename=None, save=False):

        y_cut = self.y[int(len(self.y)*0):int(len(self.y)*1)]

        self.zcr = librosa.feature.zero_crossing_rate(y_cut, hop_length=int(len(y_cut)/10))[0]

        ZCRBi = np.expand_dims(self.zcr, axis=0)

        if graph or save:

            plt.figure(figsize=(10, 4))
            vmin = 0
            vmax = 1
            plt.imshow(ZCRBi, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(label='ZCR')
            plt.title(f'ZCR - {filename}')
            plt.xlabel('Tiempo (ventanas)')
            plt.ylabel('Frecuencia (Hz)')

        if graph:
            plt.show()
            plt.close()
        if save:
            plt.savefig(f"./props/ZCR/zeroCrossingRates_of_file_{filename}.png")
            plt.close()

    def calcDuration(self, graph=False, filename=None, save=False):

        self.duration = len(self.y) / self.sr

    def temporalEvolve(self, graph=False, filename=None, save=False):

        y_rect = np.abs(self.y)

        window_size = 3000

        y_smoothed = np.convolve(y_rect, np.ones(window_size) / window_size, mode='same')

        y_normalized = (y_smoothed - np.min(y_smoothed)) / (np.max(y_smoothed) - np.min(y_smoothed))

        y_smoothed2 = np.convolve(y_normalized, np.ones(window_size) / window_size, mode='same')

        if graph or save:

            plt.figure(figsize=(10, 6))
            plt.plot(y_smoothed)
            plt.plot(y_normalized)
            plt.plot(y_smoothed2)
            plt.xlabel('Tiempo (muestras)')
            plt.ylabel('Amplitud')
            plt.title(f'Envolvente Temporal - {filename}')
            plt.legend(["Envolvente Temporal RS", "Envolvente Temporal RSN", "Envolvente Temporal RSNS"])

        if graph:

            plt.show()
            plt.close()

        if save:

            plt.savefig(f"./props/temporalEvolve/temporalEvolve_of_file_{filename}.png")
            plt.close()

    def spectralEvolve(self, graph=False, filename=None, save=False):

        # Gráfico de evolución espectral

        D = librosa.stft(self.y)
        magnitude = librosa.amplitude_to_db(abs(D), ref=np.max)

        freqs = librosa.fft_frequencies(sr=self.sr)

        spectral = np.mean(magnitude, axis=1)

        window_size = 10

        spectral = np.convolve(spectral, np.ones(window_size) / window_size, mode='same')

        hop_length = int(len(self.y) / 10)

        # Gráfico de centroides espectrales

        self.spectralCentroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, hop_length=hop_length)[0]

        self.promCentroids1 = (self.spectralCentroid[4] + self.spectralCentroid[5] + self.spectralCentroid[6]) / 3

        spectralCentroidBi = np.expand_dims(self.spectralCentroid, axis=0)

        if graph or save:

            plt.figure(figsize=(10, 4))
            vmin = 600
            vmax = 6000
            plt.imshow(spectralCentroidBi, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Energía')
            plt.title(f'Evolución Espectral - {filename}')
            plt.xlabel('Tiempo (ventanas)')
            plt.ylabel('Frecuencia (Hz)')

            if graph:

                plt.show()
                plt.close()

            if save:

                plt.savefig(f"./props/spectralCentroids/spectralCentroids_of_file_{filename}.png")
                plt.close()

        if graph or save:

            plt.figure(figsize=(10, 4))
            plt.plot(freqs, spectral)
            plt.xlabel('Frecuencia (Hz)')
            plt.ylabel('Amplitud')
            plt.title('Curva de Amplitud vs Frecuencia')
            plt.legend(["Curva de Amplitud vs Frecuencia Suavizada"])

            if graph:

                plt.show()
                plt.close()

            if save:

                plt.savefig(f"./props/spectralEnvelope/spectralEnvelope_of_file_{filename}.png")
                plt.close()

    # Métodos de usuario

    def getData(self):

        returnPackage = []
        returnPackage.append(self.duration)
        returnPackage.append(self.promCentroids1)
        returnPackage.extend(self.zcr)

        return returnPackage

    def calculateDistance(self, allAudioData, data, N, graph=False, save=False, filename=None):

        N = N
        XY = []
        distances = []

        data = np.array(data)
        datasetAudioFeatures = np.array([fruit["data"] for fruit in allAudioData])

        min_vals = np.min(datasetAudioFeatures, axis=0)
        max_vals = np.max(datasetAudioFeatures, axis=0)
        data = (data - min_vals) / (max_vals - min_vals)
        datasetAudioFeatures = (datasetAudioFeatures - min_vals) / (max_vals - min_vals)

        weights = np.array([1, 2, 0, .55, 1, 1, 2, 2, 2, 2, 2, 1, 1])

        while len(weights) < len(data):
            weights = np.append(weights, 1)

        while len(weights) > len(data):
            weights = np.delete(weights, -1)

        data = data * weights
        datasetAudioFeatures = datasetAudioFeatures * weights

        c = 0
        for audioData in datasetAudioFeatures:
            distance = np.linalg.norm(data - audioData)
            distances.append({"fruit": allAudioData[c]["fruit"], "distance": distance})
            c += 1

        distances = sorted(distances, key=lambda x: x["distance"])
        distances = distances[:N]
        nearestFruits = [i["fruit"] for i in distances]

        if graph or save:

            pca = PCA(n_components=2)
            all_data_pca = pca.fit_transform(np.vstack((datasetAudioFeatures, data)))

            plt.figure(figsize=(8, 6))
            for i, audioData in enumerate(all_data_pca[:-1]):

                if allAudioData[i]["fruit"] == "manzana":
                    color = 'red'
                    marker = 'o'
                elif allAudioData[i]["fruit"] == "banana":
                    color = 'yellow'
                    marker = 'o'
                elif allAudioData[i]["fruit"] == "pera":
                    color = 'green'
                    marker = 'o'
                else:
                    color = 'orange'
                    marker = 'o'

                plt.scatter(audioData[0], audioData[1], label=None, c=color, marker=marker)

            plt.scatter(all_data_pca[-1, 0], all_data_pca[-1, 1], label=None, c='black', marker='o')

            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.title(f'Gráfico de dispersión PCA en 2D - {filename}')

        if graph:
            plt.show()
            plt.close()

        if save:
            plt.savefig(f"./tests/knnMap/pca_of_file_{filename}.png")
            plt.close()

        return nearestFruits
