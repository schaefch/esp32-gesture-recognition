/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_

#include "driver/i2c.h"
#include "hal/i2c_ll.h"
#include <string>
#include <vector>

// Pins
constexpr auto SCL{27};
constexpr auto SDA{14};
// Constants
#define I2C_MASTER_NUM I2C_NUMBER(CONFIG_I2C_MASTER_PORT_NUM)
constexpr auto ADDR{0x29};
constexpr auto WRITE_BIT{I2C_MASTER_WRITE};
constexpr auto READ_BIT{I2C_MASTER_READ};
constexpr auto ACK_CHECK_EN{0x01};
constexpr auto ACK_CHECK_DIS{0x00};
constexpr auto ACK_VAL{0x00};
constexpr auto NACK_VAL{0x01};

// Model related
const std::vector<std::string> labels = {"idle", "0", "1", "2", "3", "4",
                                         "5",    "6", "7", "8", "9"};

constexpr auto count_axis{3};
constexpr auto count_samples{30};
const auto count_categories{labels.size()};

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.
void setup();

// Runs one iteration of data gathering and inference. This should be called
// repeatedly from the application code. The name needs to be loop() for Arduino
// compatibility.
void loop();

#ifdef __cplusplus
}
#endif

#endif // TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MAIN_FUNCTIONS_H_
