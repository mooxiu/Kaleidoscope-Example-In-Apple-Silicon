# LLVM Kaleidoscope Example For Apple Silicon

This mainly contains my configuration for the project to work on my Apple Silicon Macbook (Macbook M1Pro 2021).

The original example can be found in [llvm-project](https://github.com/llvm/llvm-project), but it replies on other
documents in the project, to run it is pretty troublesome. I want to keep a minimal version.

The code is written following [the tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html), but
there might be some minor diffs between my code and the tutorial and the example code in llvm-project repository.

## Prerequisite

- Install LLVM:
    - although there's a built-in LLVM in Mac, but it does not
      include some necessary parts.
        - Install through homebrew:
          ```shell
            brew install llvm
          ```
    - You probably need to configure path for the compiler to find LLVM, but I have `CMakeLists.txt` to do that for you.

## Build

```shell
mkdir build && cd build
cmake ./..
make
```

After any change, just go to `build` directory and just run `make`.

## Q & A

- What is the `include` dir?
    - It contains some necessary classes for the project. I just copied from the original LLVM example.
