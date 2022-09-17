# docker build -t insilica/biobricks:3.0-cuda .
# docker push insilica/biobricks:3.0-cuda
# docker run --gpus all -it -v $(pwd):/biosim insilica/biobricks:3.0-cuda /bin/bash
# you need nvidia-docker2 to run this container 
#    see the install instructions 
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
FROM rocker/ml-verse:cuda11.1

RUN apt-get update -y
RUN apt-get -y --no-install-recommends install git libxml2-dev

RUN apt-get install -y python3 python3-pip
RUN pip install dvc
RUN pip install dvc-s3

# Biobricks dependencies
RUN install2.r --error arrow
RUN install2.r --error dplyr
RUN install2.r --error fs
RUN install2.r --error purrr
RUN install2.r --error xml2
RUN install2.r --error readr
RUN install2.r --error rvest
RUN install2.r --error DBI
RUN install2.r --error RSQLite
RUN install2.r --error yaml
RUN install2.r --error pacman

# Biobricks testing
RUN install2.r --error remotes
RUN Rscript -e "remotes::install_github('biobricks-ai/biobricks-r')"
RUN Rscript -e "remotes::install_github('biobricks-ai/bricktools')"

# create 'r-reticulate' conda environment w/ tensorflow
RUN Rscript -e "reticulate::install_miniconda()"
RUN Rscript -e "reticulate::conda_install(packages = 'python=3.8.10')"

# machine learning libraries
RUN install2.r --error keras
RUN install2.r --error tensorflow
RUN Rscript -e "tensorflow::install_tensorflow()"

CMD ["/bin/bash"]