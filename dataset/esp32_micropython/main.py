from machine import Pin, I2C
import time
import struct

scl = Pin(27)
sda = Pin(14)

pull1 = Pin(32, Pin.IN, Pin.PULL_UP)
pull2 = Pin(33, Pin.IN, Pin.PULL_UP)

i2c = I2C(scl=scl, sda=sda)

# Init BNO055 accel only
i2c.writeto_mem(41, 0x3D, b"\x01")

time.sleep_ms(100)

last_time = 0

while True:
    cur_time = time.time_ns()

    if cur_time > last_time + 100e6:  # 100ms
        last_time = cur_time
        readings = i2c.readfrom_mem(41, 0x08, 6)
        x, y, z = (
            struct.unpack("h", readings[i * 2 : (i + 1) * 2])[0]
            for i in range(3)
        )
        print(
            str(time.ticks_ms()) + "," + str(x) + "," + str(y) + "," + str(z)
        )
