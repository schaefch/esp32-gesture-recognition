import os
import serial
from pynput import keyboard

# Alter this to specific device
DEVICE = "/dev/ttyUSB0"
BAUDRATE = 115200

ITEMS = [
    "idle",
] + list("0123456789")

in_recording = False


def on_press(key):
    global in_recording

    if key == keyboard.Key.space:
        in_recording = True


def on_release(key):
    global stay_item
    global in_recording

    if key == keyboard.KeyCode.from_char("n"):
        stay_item = False

    if key == keyboard.Key.space:
        in_recording = False


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

with serial.Serial(DEVICE, BAUDRATE) as ser:
    _ = ser.readline()
    os.mkdir("samples")
    for item in ITEMS:
        print(f"Now recording item: {item}")
        os.mkdir(f"samples/{item}")
        counter = 0
        stay_item = True
        while stay_item:
            filename = str(counter).zfill(3) + ".csv"
            print(
                f"{item}/{filename}: Hold SPACE to collect sample. \
                  Press n to proceed to next item."
            )
            while True:
                if not stay_item:
                    break

                if in_recording:
                    break

                _ = ser.readline()

            if stay_item:
                print("Recording...", end="")
                samples = 0
                with open(f"samples/{item}/{filename}", "w") as f:
                    while in_recording:
                        f.write(ser.readline().decode("ascii"))
                        samples += 1
                print(f"Stopped. Wrote {samples} samples.")

            counter += 1
