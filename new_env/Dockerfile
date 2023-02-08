FROM yangyangfu/mpcdrl:gpu_py3
LABEL maintainer shichao(shichaoxu2023@u.northwestern.edu)

# root 
USER root

### ===============================================================================

RUN conda remove pytorch torchvision torchaudio cudatoolkit &&\
    conda install -c conda-forge pytorch-gpu &&\
    conda install seaborn &&\
    conda clean -ya


## install
RUN pip install gpytorch
