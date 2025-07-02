import board
import digitalio

class UniversalGPIO:

    def __init__(self):

        """
        Classe generica para controle de GPIO.
        """
        self.pins_names = {"DANGER": "D27",
                           "CRITICAL": "D28"}

        self.pins = {} #Armazena os GPIOs
        for name, pin in self.pins_names.items():
            gpio_pin = digitalio.DigitalInOut(getattr(board, pin))
            gpio_pin.direction = digitalio.Direction.OUTPUT
            self.pins[name] = gpio_pin
            
        print(f" GPIO inicializado: {self.pins.keys()}")

    def set_state(self, name, state):
        """
            Define o estado do GPIO

            :param name: Nome do pino
            :param state; True (HIGH) ou False (LOW)
        """
        if name in self.pins:
            self.pins[name].value = state
            print(f"{name} {'Ativado' if state else 'Desativado'}")
        else:
            print(f" ERRO: o pino '{name}' não está configurado!")

    def set_critical(self):
        self.set_state("CRITICAL", True)

    def set_danger(self):
        self.set_state("DANGER", True)

    def reset_critical(self):
        self.set_state("CRITICAL", False)

    def reset_danger(self):
        self.set_state("DANGER", False)

    def cleanup(self):

        for pin in self.pins.values():
            pin.value = False
        print("GPIOs Resetados") 