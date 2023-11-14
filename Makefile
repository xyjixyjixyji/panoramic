.PHONY: build
build: configure
	cmake --build build

.PHONY: configure
configure:
	cmake -B build

.PHONY: clean
clean:
	rm -rf build
