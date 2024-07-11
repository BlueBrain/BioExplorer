# Copyright 2020 - 2023 Blue Brain Project / EPFL
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

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Async++ as a shared library" FORCE)

add_subdirectory(async++)
if(APPLE)
  # std::exception derivatives are not catched otherwise, e.g. in unit tests for
  # testing loading errors; see https://stackoverflow.com/questions/21737201
  target_compile_options(Async++ PRIVATE -frtti)
endif()
list(APPEND LCOV_EXCLUDE '${PROJECT_SOURCE_DIR}/deps/async++/include/async++/*'
                         '${PROJECT_SOURCE_DIR}/deps/async++/src/*')
