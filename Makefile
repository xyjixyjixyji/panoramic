CPP_SRC = $(shell find . -type f -name "*.hpp" -o -name "*.cpp" -o -name "*.cu")

.PHONY: build
build: configure
	cmake --build build

.PHONY: configure
configure:
	cmake -B build -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++

.PHONY: format
format:
	clang-format --style=file -i $(CPP_SRC)

.PHONY: clean
clean:
	rm -rf build
