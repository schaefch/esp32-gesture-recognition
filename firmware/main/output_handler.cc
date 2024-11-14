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

#include "output_handler.h"
#include "main_functions.h"
#include "tensorflow/lite/micro/micro_log.h"
#include <string>
#include <vector>

void HandleOutput(float *values, size_t count) {

  static std::vector<unsigned int> counters;

  static int last_cmd = 0;

  constexpr float threshold = 0.90f;

  constexpr float threshold_cnt = 7;

  int max_i = 0;

  counters.resize(count);

  for (auto i = 0; i < count; i++) {

    if (values[i] >= threshold) {

      counters[i]++;

    } else {
      counters[i] = 0;
    }

    if (counters[i] >= threshold_cnt) {
      MicroPrintf(labels[i].c_str());
    }
  }

  if (max_i != last_cmd) {
    last_cmd = max_i;
  }
}
