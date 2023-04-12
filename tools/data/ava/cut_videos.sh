#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# Cut each video from its 15th to 30th minute.

IN_DATA_DIR='../../../../../_video/5.mp4'
OUT_DATA_DIR='../../../../../video/5_1.mp4'

out_name="${OUT_DATA_DIR}"
if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 3 -t 1 -i "${IN_DATA_DIR}" -r 6 -strict experimental "${out_name}"
fi

