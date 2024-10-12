To install the `face_recognition` library using **conda**, you'll first need to ensure that some dependencies, like `dlib` and `cmake`, are installed correctly because `face_recognition` relies on `dlib` for its underlying face recognition algorithms.

Here’s a step-by-step guide to install `face_recognition` with **conda**:

### 1. Create a new conda environment (optional)
It's recommended to create a separate environment to avoid conflicts with existing packages.

```bash
conda create -n face_env python=3.8
conda activate face_env
```

### 2. Install dependencies

#### Step 2.1: Install `cmake`
`dlib` requires `cmake` to build the package, so install it via conda:

```bash
conda install -c conda-forge cmake
```

#### Step 2.2: Install `boost` and other required libraries
`boost` is a requirement for `dlib`, and sometimes you might also need `libjpeg` and other packages for image handling.

```bash
conda install -c conda-forge boost libjpeg
```

### 3. Install `dlib`
The `face_recognition` library relies on `dlib`, so you must install it first. You can install `dlib` via conda-forge:

```bash
conda install -c conda-forge dlib
```

### 4. Install `face_recognition`
Once `dlib` is installed, you can install `face_recognition` using `pip` inside the conda environment:

```bash
pip install face_recognition
```

Unfortunately, `face_recognition` is not available directly through conda channels, but it works well when installed via `pip` after installing `dlib` through conda.

### 5. Verify the Installation
You can verify that everything is installed correctly by running a small Python script:

```bash
python -c "import face_recognition; print(face_recognition.__version__)"
```

If it outputs the version number without errors, the installation was successful.

### Quick Summary of Commands:
```bash
# Create and activate conda environment (optional)
conda create -n face_env python=3.8
conda activate face_env

# Install cmake
conda install -c conda-forge cmake

# Install boost and other required libraries
conda install -c conda-forge boost libjpeg

# Install dlib
conda install -c conda-forge dlib

# Install face_recognition via pip
pip install face_recognition

# Verify installation
python -c "import face_recognition; print(face_recognition.__version__)"
```

This setup should give you a working installation of `face_recognition` in a conda environment.