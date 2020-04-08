## Prerequisites

* miniconda3-4.3.30
  * You can install Miniconda from https://conda.io/miniconda.html
  * If you would like to manage multiple Python versions on your machine at once, I recommend using pyenv (https://github.com/pyenv/pyenv)



## Folder Structure

The folder structure follows the suggestions made by William Stafford Noble in *A Quick Guide to Organizing Computational Biology Projects*.

Noble, William Stafford. “A Quick Guide to Organizing Computational Biology Projects.” *PLoS Computational Biology*, vol. 5, no. 7, 2009, doi:10.1371/journal.pcbi.1000424.

https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1000424&type=printable



## Setup

Navigate the the root of the project folder.

Create a conda environment using the requirements in the environment file:

```bash
conda env create -f requirements.yml
```

Activate the conda environment:

```bash
source activate covid-19
```





## Teardown

Navigate to the root of the project folder.

Remove the conda environment:

```bash
conda remove --name covid-19 --all
```

To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run:

```
conda info --envs
```

You are now free to delete the ```covid-19``` folder.

