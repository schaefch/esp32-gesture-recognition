/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "output_handler.h"

#include <queue>
#include <vector>

#include "driver/i2c.h"
#include "hal/i2c_ll.h"

const i2c_config_t conf = {
    .mode = I2C_MODE_MASTER,
    .sda_io_num = SDA,
    .scl_io_num = SCL,
    .sda_pullup_en = GPIO_PULLUP_ENABLE,
    .scl_pullup_en = GPIO_PULLUP_ENABLE,
    .master =
        {
            .clk_speed = 100000,
        },
};

namespace {
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
float *model_input_buffer = nullptr;

constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

} // namespace

std::deque<float> samples;

void setup() {

  samples.resize(count_axis * count_samples);
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  constexpr auto count_different_ops{8};

  static tflite::MicroMutableOpResolver<count_different_ops> resolver;

  if (resolver.AddConv2D() != kTfLiteOk) {
    return;
  }

  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }

  if (resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }

  if (resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  if (resolver.AddExpandDims() != kTfLiteOk) {
    return;
  }

  if (resolver.AddAdd() != kTfLiteOk) {
    return;
  }

  if (resolver.AddMul() != kTfLiteOk) {
    return;
  }

  if (resolver.AddMean() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  model_input_buffer = tflite::GetTensorData<float>(input);

  // Setup I2C Interface to BNO055
  esp_err_t err = i2c_param_config(I2C_NUM_0, &conf);

  i2c_set_timeout(I2C_NUM_0, I2C_LL_MAX_TIMEOUT);

  if (err != ESP_OK) {
    MicroPrintf("I2C conf not ok");
  }
  ESP_ERROR_CHECK(i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0));

  i2c_cmd_handle_t cmd = i2c_cmd_link_create();
  i2c_master_start(cmd);
  i2c_master_write_byte(cmd, (ADDR << 1) | I2C_MASTER_WRITE,
                        1 /* expect ack */);

  std::vector<uint8_t> txbuf{0x3d, 0x01};
  i2c_master_write(cmd, &txbuf.front(), txbuf.size(), true);
  i2c_master_stop(cmd);

  constexpr auto ticks_for_communication{300};

  ESP_ERROR_CHECK(i2c_master_cmd_begin(
      I2C_NUM_0, cmd, ticks_for_communication / portTICK_PERIOD_MS));

  i2c_cmd_link_delete(cmd);
}

// The name of this function is important for Arduino compatibility.
void loop() {

  i2c_cmd_handle_t cmd = i2c_cmd_link_create();
  i2c_master_start(cmd);
  i2c_master_write_byte(cmd, (ADDR << 1) | I2C_MASTER_WRITE,
                        1 /* expect ack */);

  // Get accelerometer data
  i2c_master_write_byte(cmd, 0x08, true);
  i2c_master_start(cmd);
  i2c_master_write_byte(cmd, (ADDR << 1) | I2C_MASTER_READ, 1 /* expect ack */);

  // Read
  std::vector<int16_t> rxbuf{0, 0, 0};
  i2c_master_read(cmd, ((uint8_t *)&rxbuf.front()), 2 * rxbuf.size(),
                  I2C_MASTER_LAST_NACK);
  i2c_master_stop(cmd);

  constexpr auto ticks_for_communication{800};

  ESP_ERROR_CHECK(i2c_master_cmd_begin(
      I2C_NUM_0, cmd, ticks_for_communication / portTICK_PERIOD_MS));

  i2c_cmd_link_delete(cmd);

  // Enqueue data
  for (auto i = 0; i < count_axis; i++) {
    samples.pop_front();
    samples.push_back(rxbuf[i]);
  }

  for (auto nr = 0; nr < count_samples; nr++) {
    for (auto axis = 0; axis < count_axis; axis++) {
      model_input_buffer[count_axis * nr + axis] =
          samples[count_axis * nr + axis];
    }
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");

    return;
  }

  HandleOutput(tflite::GetTensorData<float>(output), count_categories);
}
