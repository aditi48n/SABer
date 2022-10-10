# SABer

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1a2954edef114b81a583bb23ffba2ace)](https://app.codacy.com/gh/hallamlab/SABer?utm_source=github.com&utm_medium=referral&utm_content=hallamlab/SABer&utm_campaign=Badge_Grade_Dashboard)

SAG Anchored Binner for recruiting metagenomic reads using single-cell amplified genomes as references

Check out the [wiki](https://github.com/hallamlab/SABer/wiki) for tutorials and more information on SABer!!

### Install SABer and Dependencies
Currently the easiest way to install SABer is to use a conda virtual environment.  
This will require the installation of [Anaconda](https://www.anaconda.com/distribution/).  
Once Anaconda is installed, you can follow the directions below to install all dependencies and SABer within a conda environment.
```sh
git clone https://github.com/hallamlab/SABer.git
cd SABer
```
 Now use `make` to create the conda env, activate it, and install SABer via pip.
```sh
make install-saberenv
conda activate saber_cenv
make install-saber
```

### Docker and Singularity containers
If you would like to use a docker or singularity container of SABer they are available:

[Docker](not_available_yet)

[Singularity](not_available_yet)

They can also be build from scratch using the following commands:

Docker:
```sh
make docker-build
```
Singularity:
```sh
make singularity-local-build
``` 
