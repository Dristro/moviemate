#!/bin/bash

echo "Setup moviemate"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.11 >="
    exit 1
fi

echo "Python found: $(python3 --version)"
echo ""

# Ask user for environment preference
echo "Choose environment setup method:"
echo "1) uv (fast, recommended)"
echo "2) pip (current environment)"
echo "3) conda (create new environment)"
echo ""

read -p "Enter your choice (1/2/3): " ENV_CHOICE
echo ""

# Install dependencies based on choice
case $ENV_CHOICE in
    1)
        echo "Using uv..."
        if ! command -v uv &> /dev/null; then
            echo "uv is not installed. Install it from https://github.com/astral-sh/uv"
            exit 1
        fi

        uv venv
        source .venv/bin/activate
        uv sync
        ;;

    2)
        echo "Using pip (current environment)..."
        pip install -r requirements.txt
        ;;

    3)
        echo "Using conda..."
        if ! command -v conda &> /dev/null; then
            echo "Conda is not installed. Please install Miniconda/Anaconda."
            exit 1
        fi

        read -p "Enter new conda env name: " ENV_NAME
        conda create -y -n "$ENV_NAME" python=3.10
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"

        pip install -r requirements.txt
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Check install success
if [ $? -ne 0 ]; then
    echo "Failed to install deps"
    exit 1
fi

echo "Deps installed!"
echo ""


echo ""
echo "Setup complete!"
echo ""
echo "To start the application, run:"
echo "  streamlit run app/main.py"
