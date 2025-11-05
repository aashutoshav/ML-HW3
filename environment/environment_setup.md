# Environment Setup

1. Refer to the EdStem post about [VS Code and Anaconda Setup Guide](https://edstem.org/us/courses/81465/discussion/6899817) if you do not already have an Anaconda distribution installed (if you know what you're doing with regard to python, this TA recommends Miniconda or Miniforge; Anaconda just has a lot of overhead).
2. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Use the command: `conda env create -f environment/ml_hw3_env_<OS>.yml`
    - Replace `<OS>` with what's relevant to you: `[linux_64, osx_64, osx_arm64, win_64]`.
    - This should create an environment named `ml-hw3`.
3. Activate the conda environment:
    - Use the command: `conda activate ml-hw3`
    - You should see (ml-hw3) beside your cursor where (base) was before.
4. Activating the environment does not mean the environment will be used when running Python applications. In VS Code, follow these steps:
    - Open the Jupyter notebook file
    - On the top right, click the button that states "Select Kernel"
    - Select "Python Environments..."
    - Select the ml-hw3 environment

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
