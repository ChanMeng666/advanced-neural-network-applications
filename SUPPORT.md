# Support

## Getting Help

If you need help with this project, here are the available channels:

### Questions and Discussions

- **GitHub Issues**: For bug reports and feature requests, please [open an issue](https://github.com/ChanMeng666/advanced-neural-network-applications/issues)
- **GitHub Discussions**: For general questions, ideas, and community conversations, visit the [Discussions](https://github.com/ChanMeng666/advanced-neural-network-applications/discussions) tab

### Common Issues

**Notebooks fail to load data files**
- Make sure you launch Jupyter from the project root directory
- Notebooks expect data files at `../data/` relative to the `notebooks/` directory
- On Google Colab, the first cell automatically downloads the required data

**Missing dependencies**
- Run `pip install -r requirements.txt` to install all required packages
- If using Conda: `conda install numpy pandas matplotlib scikit-learn jupyter`

**3D plots not rendering**
- Ensure Matplotlib is installed with its default backend
- In JupyterLab, you may need to install `ipympl` for interactive 3D plots

### Run Online (No Installation)

If you prefer not to install anything locally:

- **Binder**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ChanMeng666/advanced-neural-network-applications/main?labpath=notebooks)
- **Google Colab**: Open any notebook directly — see links in [README](README.md#notebooks)

## Scope of Support

This is an open-source educational project maintained on a volunteer basis. We do our best to respond to issues, but cannot guarantee response times. Please be patient and respectful.

## Security Issues

For security-related concerns, **do not open a public issue**. Instead, follow the process described in [SECURITY.md](SECURITY.md).
