CPP_SRC = $(shell find . -type f -name "*.hpp" -o -name "*.cpp" -o -name "*.cu")

.PHONY: build
build: configure
	cmake --build build

.PHONY: configure
configure:
	cmake -B build

.PHONY: format
format:
	clang-format --style=file -i $(CPP_SRC) -n --Werror

.PHONY: clean
clean:
	rm -rf build

