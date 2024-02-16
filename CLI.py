import time
from cmd import Cmd
import sys
import matplotlib.pyplot as plt
import signal

from Controller import Controller


LGREEN = '\033[92m'
LBLUE = '\033[94m'
LRED = '\033[91m'
RESET = '\033[0m'
LYELLOW = '\033[93m'


class CLI(Cmd):

    doc_header = "Ayuda de comandos documentados"
    undoc_header = "Ayuda de comandos no documentados"
    ruler = "="

    def __init__(self):
        super().__init__()
        self.completekey = "tab"
        self.controller = Controller(self.callback)
        print(f"{LGREEN} --> {RESET}Iniciando entrada de comandos. Usar help para ver comandos.")

# Métodos de usuario

    def do_generarConocimiento(self, args):

        if args == "s":
            args = True
        elif args == "n":
            args = False

        if args == "":
            try:
                self.controller.generateKnowledge()
            except Exception as e:
                self.callback(1, "Error al generar conocimiento. Ejecución detenida.")
                self.callback(1, e)

        else:
            try:
                self.controller.generateKnowledge(args)
            except Exception as e:
                self.callback(1, "Error al generar conocimiento. Ejecución detenida.")
                self.callback(1, e)

    def do_audioTest(self, args):
        """Test de audio."""

        self.controller.audioTest()

    def do_getFruit(self, args):
        """Obtener fruta."""

        if args == "":
            try:
                self.controller.getFruit()
            except Exception as e:
                self.callback(1, "Error al obtener fruta. Ejecución detenida.")
                self.callback(1, e)

        else:
            try:
                self.controller.getFruit(args)
            except Exception as e:
                self.callback(1, "Error al obtener fruta. Ejecución detenida.")
                self.callback(1, e)


# Métodos de interacción

    def do_exit(self, args):
        """Salir de la CLI."""
        print(f"{LGREEN} --> {RESET}Cerrando CLI...")

        return True

    def do_help(self, args):
        """Muestra la ayuda de los comandos disponibles."""

        print(f"{LGREEN}===== {RESET}Comandos disponibles{LGREEN} ====={RESET}")

        for attr in dir(self):
            if attr[:3] == "do_":
                print(f"{LGREEN} --> {RESET}{attr[3:]}")

    def default(self, args):
        """Comando por defecto."""
        print(f"{LRED} --> {RESET}Comando no reconocido.")

    def callback(self, status, message):
        if status == 0:
            print(f"{LGREEN} --> {RESET}{message}")
        elif status == 1:
            print(f"{LRED} --> {RESET}{message}")

