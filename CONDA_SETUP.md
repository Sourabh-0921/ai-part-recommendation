# Conda Setup Guide for AI Parts Recommendation System

This guide provides detailed instructions for setting up the AI Parts Recommendation System using conda for package management.

## Prerequisites

### 1. Install Conda

If you don't have conda installed, choose one of these options:

#### Option A: Anaconda (Full Distribution)
- Download from: https://www.anaconda.com/products/distribution
- Includes conda, Python, and many pre-installed packages
- Larger download (~3GB) but includes everything you need

#### Option B: Miniconda (Minimal Distribution)
- Download from: https://docs.conda.io/en/latest/miniconda.html
- Minimal installation with just conda and Python
- Smaller download (~400MB), install packages as needed

### 2. Verify Installation

```bash
# Check conda version
conda --version

# Check Python version
python --version
```

## Environment Setup

### 1. Automated Setup (Recommended)

```bash
# Navigate to project directory
cd /path/to/ai-parts-recommendation-system

# Run the automated setup script
./activate_conda.sh
```

This script will:
- Create the conda environment from `environment.yml`
- Install all required packages
- Activate the environment
- Provide next steps

### 2. Manual Setup

If you prefer manual control:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ai-parts-recommendation

# Verify environment is active
conda info --envs
```

## Environment Management

### Activating the Environment

```bash
# Activate the environment
conda activate ai-parts-recommendation

# Verify it's active (should show in prompt)
echo $CONDA_DEFAULT_ENV
```

### Deactivating the Environment

```bash
# Deactivate current environment
conda deactivate
```

### Updating the Environment

```bash
# Update environment from environment.yml
conda env update -f environment.yml

# Or update specific packages
conda update pandas numpy scikit-learn
```

### Removing the Environment

```bash
# Remove the environment (if needed)
conda env remove -n ai-parts-recommendation
```

## Package Management

### Installing Additional Packages

```bash
# Install a new package
conda install package_name

# Install from conda-forge channel
conda install -c conda-forge package_name

# Install via pip (for packages not available in conda)
pip install package_name
```

### Updating Packages

```bash
# Update all packages in environment
conda update --all

# Update specific package
conda update package_name
```

### Listing Packages

```bash
# List all packages in current environment
conda list

# List packages for specific environment
conda list -n ai-parts-recommendation

# Export current environment
conda env export > environment.yml
```

## Development Workflow

### 1. Daily Development

```bash
# Activate environment
conda activate ai-parts-recommendation

# Work on your code
# ...

# Deactivate when done
conda deactivate
```

### 2. Running the Application

```bash
# Activate environment
conda activate ai-parts-recommendation

# Start the API
python run_api.py

# Or run tests
pytest
```

### 3. Jupyter Notebook Development

```bash
# Activate environment
conda activate ai-parts-recommendation

# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## Troubleshooting

### Common Issues

#### 1. Environment Not Found
```bash
# List all environments
conda env list

# If environment doesn't exist, recreate it
conda env create -f environment.yml
```

#### 2. Package Conflicts
```bash
# Check for conflicts
conda list --show-channel-urls

# Resolve conflicts by updating
conda update --all
```

#### 3. Slow Package Installation
```bash
# Use conda-forge channel (faster)
conda install -c conda-forge package_name

# Or use mamba (faster conda)
conda install mamba
mamba install package_name
```

#### 4. Environment Activation Issues
```bash
# Initialize conda for your shell
conda init bash  # for bash
conda init zsh   # for zsh
conda init fish  # for fish

# Restart your terminal after running conda init
```

### Performance Optimization

#### 1. Use Mamba (Faster Package Manager)
```bash
# Install mamba
conda install mamba

# Use mamba instead of conda
mamba env create -f environment.yml
mamba activate ai-parts-recommendation
```

#### 2. Optimize Conda Configuration
```bash
# Configure conda for better performance
conda config --set channel_priority strict
conda config --set show_channel_urls true
```

## Environment File Structure

The `environment.yml` file contains:

- **name**: Environment name (`ai-parts-recommendation`)
- **channels**: Package sources (conda-forge, defaults)
- **dependencies**: All required packages
- **pip dependencies**: Packages not available in conda

### Key Dependencies

- **Python**: 3.9
- **ML Libraries**: pandas, numpy, scikit-learn, lightgbm
- **API Framework**: fastapi, uvicorn, pydantic
- **Database**: sqlalchemy, psycopg2, alembic
- **Development**: pytest, black, flake8, jupyter

## Best Practices

### 1. Environment Isolation
- Always use conda environments for different projects
- Don't install packages in the base environment
- Use descriptive environment names

### 2. Package Management
- Prefer conda packages over pip when available
- Use conda-forge channel for better package selection
- Pin package versions in production environments

### 3. Environment Sharing
- Export environment with exact versions: `conda env export > environment.yml`
- Include both conda and pip dependencies
- Document any manual installation steps

### 4. Development Workflow
- Activate environment before starting work
- Use environment-specific tools (jupyter, pytest)
- Deactivate environment when switching projects

## Integration with IDEs

### VS Code
1. Install Python extension
2. Select conda environment as interpreter
3. Command Palette → "Python: Select Interpreter"
4. Choose `ai-parts-recommendation` environment

### PyCharm
1. Go to Settings → Project → Python Interpreter
2. Add New Interpreter → Conda Environment
3. Select existing environment or create new one

### Jupyter
```bash
# Install ipykernel in the environment
conda activate ai-parts-recommendation
conda install ipykernel

# Register environment with Jupyter
python -m ipykernel install --user --name ai-parts-recommendation
```

## Production Deployment

### 1. Environment Export
```bash
# Export exact environment for production
conda env export > environment-production.yml

# Remove build strings for cleaner export
conda env export --no-builds > environment-production.yml
```

### 2. Docker Integration
```dockerfile
# Use conda in Docker
FROM continuumio/miniconda3

# Copy environment file
COPY environment.yml .

# Create environment
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "ai-parts-recommendation", "/bin/bash", "-c"]
```

## Support

For conda-specific issues:
- Conda Documentation: https://docs.conda.io/
- Conda Forge: https://conda-forge.org/
- Stack Overflow: Tag questions with `conda`

For project-specific issues:
- Check the main README.md
- Review the API documentation
- Contact the development team
