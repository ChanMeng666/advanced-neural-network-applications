# Contributing to Advanced Neural Network Applications

Thank you for your interest in contributing! This guide explains how to get involved.

## How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/ChanMeng666/advanced-neural-network-applications/issues/new?template=bug_report.md) with the following information:

- Which notebook you were running
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your Python version and operating system

### Suggesting Features

Have an idea for a new notebook, model, or visualization? [Open a feature request](https://github.com/ChanMeng666/advanced-neural-network-applications/issues/new?template=feature_request.md) describing:

- The problem or learning gap you want to address
- Your proposed solution
- Any references or resources

### Submitting Changes

1. **Fork** the repository
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/advanced-neural-network-applications.git
   cd advanced-neural-network-applications
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Make your changes** and verify that all notebooks run without errors
6. **Commit** with a clear message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add multi-layer perceptron notebook"
   ```
7. **Push** and open a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone and install
git clone https://github.com/ChanMeng666/advanced-neural-network-applications.git
cd advanced-neural-network-applications
pip install -r requirements.txt

# Launch notebooks
jupyter lab
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Include clear comments explaining mathematical steps
- Use descriptive variable names (e.g., `learning_rate` not `lr`)
- Add markdown cells to explain concepts before code cells

## Commit Message Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix       | Use case                          |
|--------------|-----------------------------------|
| `feat:`      | New notebook, model, or feature   |
| `fix:`       | Bug fix                           |
| `docs:`      | Documentation changes             |
| `refactor:`  | Code restructuring                |
| `chore:`     | Maintenance tasks                 |

## Pull Request Guidelines

- Reference any related issues (e.g., `Closes #12`)
- Describe what your PR does and why
- Ensure all notebooks execute from top to bottom without errors
- Include screenshots for new visualizations

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to ChanMeng666@outlook.com.
