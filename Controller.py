import os
import json
import datetime
import sounddevice
import wave
from math import ceil

from Knn import Knn

class Controller:

    def __init__(self, callback):
        self.knn = Knn()
        self.allAudioData = []
        self.callback = callback
        pass

    def generateKnowledge(self, genProps=True):

        self.allAudioData = []

        for fruit in os.listdir("dataset/audio"):
            for file in os.listdir(f"dataset/audio/{fruit}"):

                self.callback(0, f"Extrayendo datos de {file}...")

                filename = file.replace(".wav", "")

                file_path = f"dataset/audio/{fruit}/{file}"
                self.knn.extractAudioData(file_path, graph=False, save=genProps)
                data = self.knn.getData()

                self.allAudioData.append({"fruit": fruit, "data": data})

        with open("./data/audioData.json", "w") as f:
            json.dump(self.allAudioData, f)

    def audioTest(self):

        if len(self.allAudioData) == 0:
            with open("./data/audioData.json", "r") as f:
                self.allAudioData = json.load(f)

        global_total = 0
        global_success = 0
        global_errors = 0

        for fruit in os.listdir("./tests/audioStatTest"):

            local_total = 0
            success_counter = 0
            local_errors = 0

            for file in os.listdir(f"./tests/audioStatTest/{fruit}"):

                global_total += 1
                local_total += 1

                file_path = f"./tests/audioStatTest/{fruit}/{file}"

                try:
                    self.knn.extractAudioData(file_path, graph=False, save=False)

                except Exception as e:

                    self.callback(1, e)
                    local_errors += 1
                    global_errors += 1
                    continue

                data = self.knn.getData()
                nearestNeighbors = self.knn.calculateDistance(self.allAudioData, data, 5, graph=False, save=True, filename=file.replace(".opus", ""))

                if nearestNeighbors.count(fruit) >= 3:

                    success_counter += 1
                    global_success += 1

                    self.callback(0, f"Archivo {file} reconocido como {fruit} con éxito.")

                else:

                    self.callback(1, f"Archivo {file} no reconocido como {fruit}.")

            success_percentage = success_counter / (local_total-local_errors) * 100
            self.callback(0, f"==================== Porcentaje de éxito para {fruit}: {success_percentage}% ====================")

        global_success_percentage = global_success/(global_total - global_errors) * 100
        self.callback(0, "==================================================================")
        self.callback(0, f"Porcentaje de éxito global: {global_success_percentage}%")
        self.callback(0, "==================================================================")

    def getFruit(self, filename=None):

        if filename is not None:

            filename = filename

        else:

            input_devices = sounddevice.query_devices(kind='input')
            self.callback(0, f"Dispositivos de entrada: {input_devices['name']}")

            duration = 2
            fs = 44100
            channels = 1
            dtype = 'int16'

            self.callback(0, f"Para comenzar a grabar pulsar enter. La grabación durará {duration} segundos.")
            input()
            audio = sounddevice.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=dtype)
            sounddevice.wait()

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            with wave.open(f"./tests/audioLiveTest/{filename}.wav", "wb") as f:
                f.setnchannels(channels)
                f.setsampwidth(2)
                f.setframerate(fs)
                f.writeframes(audio.tobytes())

            filename = filename + ".wav"

        try:
            self.knn.extractAudioData('./tests/audioLiveTest/' + filename, graph=True, save=False)

        except Exception as e:

            raise e

        data = self.knn.getData()

        self.callback(0, "Duración: " + str(data[0]))
        self.callback(0, "Promedio de centroides 4, 5 y 6: " + str(data[1]))

        N = 5

        if len(self.allAudioData) == 0:
            with open("./data/audioData.json", "r") as f:
                self.allAudioData = json.load(f)

        nearestFruits = self.knn.calculateDistance(self.allAudioData, data, N, graph=True, save=False, filename=None)

        if nearestFruits.count("manzana") > 2:
            nearestFruit = "manzana"
        elif nearestFruits.count("pera") > 2:
            nearestFruit = "pera"
        elif nearestFruits.count("banana") > 2:
            nearestFruit = "banana"
        else:
            nearestFruit = "naranja"

        self.callback(0, f"Frutas más cercanas: {nearestFruits}")

        self.callback(0, "==================================================================")
        if nearestFruit is not None:
            self.callback(0, f"Fruta reconocida como: {nearestFruit}")
        else:
            self.callback(0, "No se reconoció ninguna fruta.")
        self.callback(0, "==================================================================")

        #CONTINUAR EJECUCION CON PARTE DE IMAGE


